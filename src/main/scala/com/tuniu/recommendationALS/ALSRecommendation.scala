/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tuniu.recommendationALS

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel



/**
  * ALS implict recommendation based on resident time
  * Run with
  * please use `spark-submit` to submit your app.
  */
object ALSRecommendation {

  case class Params(
                     output: String = null,
                     numIterations: Int = 20,
                     lambda: Double = 1.0,
                     alpha:Double = 1.0,
                     rank: Int = 10,
                     numUserBlocks: Int = -1,
                     numProductBlocks: Int = -1,
                     implicitPrefs: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Implicit ALS") {
      head("Implicit ALS")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      opt[Double]("alpha")
        .text(s"alpha , default: ${defaultParams.alpha}")
        .action((x, c) => c.copy(alpha = x))
      opt[Int]("numUserBlocks")
        .text(s"number of user blocks, default: ${defaultParams.numUserBlocks} (auto)")
        .action((x, c) => c.copy(numUserBlocks = x))
      opt[Int]("numProductBlocks")
        .text(s"number of product blocks, default: ${defaultParams.numProductBlocks} (auto)")
        .action((x, c) => c.copy(numProductBlocks = x))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      arg[String]("<output>")
        .text("output paths " +
          "contains a transaction with each item in String and separated by a space")
        .required()
        .action{(x, c) =>
          println("out:" + x)
          c.copy(output = x)}
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --Filename.jar \
          |  --rank 5 --numIterations 20 --lambda 1.0  \
          |   /user/zhangkun6/warehouse/als_rt_pro.txt /user/zhangkun6/model/als_scala
        """.stripMargin)
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = new SparkConf()
      .setAppName(s"ALS implicit recommendation with $params")
    val sc = new SparkContext(conf)
    // 设置checkpoint存储目录，防止迭代次数过多导致stackOverFlow
    sc.setCheckpointDir("/user/zhangkun6/gbtlr/checkpoint")

    val warehouse_location = "/user/zhangkun6/warehouse"
    val spark = SparkSession.builder
      .appName(" Spark SQL Hive integration ")
      .config("spark.sql.warehouse.dir", warehouse_location)
      .enableHiveSupport()
      .getOrCreate()


    Logger.getRootLogger.setLevel(Level.WARN)

    val implicitPrefs = params.implicitPrefs

    // 表的位置,数据格式：visitor_trace, poi, rating
    /**
      * BUG记录： sql语句如果取limit 200，那么每次取的时候，值都可能不一样，导致join操作结果为空
      * 即data一直变化
      * 临时解决方案：单独创建临时表，存放要取的所有数据
      *  最终解决方案： 仅加了持久化存储方法-无效，再加上data.count() 有效，原因不明，与numData是行动操作调用了cache有关？？
       */

    val sql = "select * from tmp_zk_als_log_rt_dense"
    val data = spark.sql(sql).rdd.map(x=> (x(0), (x(1), x(2)))).persist(StorageLevel.MEMORY_AND_DISK)
    val numData = data.count()
    println(s"Got data size: $numData")

    // visitor_trace包含字符无法转换为整型，使用zipWithIndex() 映射
    val vts = data.map( x => x._1).distinct()
    val vts_with_indices = vts.zipWithIndex().cache()
    println("vts_with_indices映射关系为：")
    vts_with_indices.take(3).foreach(println)
    //  把visitor_trace设为key,并且关联起来
    val joined = data.join(vts_with_indices)


    /**
    Array[(Any, ((Any, Any), Long))]
     */

    // 生成最终数据
    val ratings_row = joined.map(x => (x._2._2, x._2._1._1,x._2._1._2))
    // 将Any类型转换为String
    val ratings_sort= ratings_row.map { x =>
      val fields = Array(x._1.toInt, x._2.toString.toInt, x._3.toString.toDouble)
      Rating(fields(0).toInt, fields(1).toInt, fields(2))
    }.cache()

    // 根据浏览时长rating进行排序并分区
    val numRatings  = ratings_sort.count()
    val partSize = (numRatings / 5).toInt
    val sortByRt = ratings_sort.sortBy(_.rating).toLocalIterator.toList
    val rt1 = sortByRt(partSize - 1).rating
    val rt2 = sortByRt(partSize * 2 - 1).rating
    val rt3 = sortByRt(partSize * 3 - 1).rating
    val rt4 = sortByRt(partSize * 4 - 1).rating

    val ratings = ratings_sort.map{ row =>
      val rtScore = row.rating match{
        case rt if rt <= rt1 => 0
        case rt if rt <= rt2 => 1
        case rt if rt <= rt3 => 2
        case rt if rt <= rt4 => 3
        case _ =>4
      }
      val result = Rating(row.user, row.product, rtScore.toDouble )
      result
    }.cache()
    ratings_sort.unpersist()

    println("The dataset has been created")

//    //以下代码是原来自设的visitor_trace映射的数据读取程序， 已作废
//    val ratings = sc.textFile(params.input).map { line =>
//      val fields = line.split('\t')
//      if (implicitPrefs) {

//        /*
//         * 隐式评分范围在 2.4-6.5之间，中位数为5.4，值越大，用户对该物品越感兴趣
//         * 将隐式评分映射为confident level: c_ui = 1+alpha*r_ui
//         * 将原来的值减去4，则新的值范围在-2.6-1.5之间，则未观测变量0可以表达为"还行","一般"
//         */
//
//        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 5)
//      } else {
//        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
//      }
//    }

    val numUsers = ratings.map(_.user).distinct().count()
    val numPois = ratings.map(_.product).distinct().count()

    println(s"Got $numRatings implicit feedbacks from $numUsers users on $numPois Pois.")


    //  注意：在隐式反馈中，测试集映射到0或1，训练集没有映射到0或1
    //  模型的调参放在ALSCrossValid中，此处拿整个样本训练
//    val splits = ratings.randomSplit(Array(0.8, 0.2))
//    println(s"training set and Test set have been split")
//    val training = splits(0).cache()
//    val test = if (params.implicitPrefs) {
//      /*
//       * 0 means "don't know" and positive values mean "confident that the prediction should be 1".
//       * Negative values means "confident that the prediction should be 0".
//       * We have in this case used some kind of weighted RMSE. The weight is the absolute value of
//       * the confidence. The error is the difference between prediction and either 1 or 0,
//       * depending on whether r is positive or negative.
//       */
//      splits(1).map(x => Rating(x.user, x.product, if (x.rating > 0) 1.0 else 0.0))
//    } else {
//      splits(1)
//    }.cache()
//
//    val numTraining = training.count()
//    val numTest = test.count()
//    println(s"Training: $numTraining, test: $numTest.")
//    ratings.unpersist(blocking = false)

    val model = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setAlpha(params.alpha)
      .setImplicitPrefs(params.implicitPrefs)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .run(ratings)

    // val rmse = computeRmse(model, test, params.implicitPrefs)

    //println(s"Test RMSE = $rmse.")

    // 保存模型
    model.save(sc, params.output)
    println("模型已保存到："+ params.output)
    data.unpersist()
    ratings.unpersist()

    // 为每个用户产生推荐
    println("为用户产生推荐")
    TopNRecommendation(sc, vts_with_indices,model)

    sc.stop()
  }


  /** Compute RMSE (Root Mean Squared Error) for testset. */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean)
  : Double = {

    /** 将预测值映射为{0，1}的范围内  */
    def mapPredictedRating(r: Double): Double = {
      if (implicitPrefs)
        //math.max(math.min(r, 1.0), 0.0)
        if (r>0) 1 else 0
      else r
    }

    // 写法1
    // 产生user Poi键 pairs
    val userPois = data.map { case Rating(user, product, rating) => (user, product) }

    // 预测user-poi pairs
    val predictions = model.predict(userPois).map { case Rating(user, product, rating) =>
      ((user, product), mapPredictedRating(rating))
    }

    // user poi 真实值  取值为0 或 1
    val ratesAndPreds = data.map { case Rating(user, product, rating) =>
      ((user, product), mapPredictedRating(rating))
    }.join(predictions)



    math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean())
  }


//    // 写法二
//    //  根据模型，预测用户对物品的的分值，predict的参数为RDD[(Int,Int)]
//    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
//
//    val ture = data.map (x => ((x.user, x.product), x.rating))
//
//    val predictionsAndRatings = predictions.map { x =>
//      ((x.user, x.product), mapPredictedRating(x.rating))
//    }.join(ture).values
//
//    // 计算RMSE值
//    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
//}
//

     //为所有的用户产生推荐
  def TopNRecommendation(sc:SparkContext,vts_with_indices:RDD[(Any, Long)],model:MatrixFactorizationModel) :Unit ={

    val indices_with_vts = vts_with_indices.map(x => (x._2.toInt, x._1.toString))
    println("indices_with_vts映射关系为：")
    indices_with_vts.take(3).foreach(println)
    // Spark加载模型,预测效率好像很低，暂时舍弃，直接用训练好的模型预测
    // val model = MatrixFactorizationModel.load(sc,path="/user/zhangkun6/warehouse/als_scala")
    // println("-------Model has been loaded!--------")
    println("---------Recommending for users-----------")
    val recommendationForAll = model.recommendProductsForUsers(num=10)
    val recForVt = recommendationForAll.join(indices_with_vts)
    val predictions = recForVt.map{
      x =>
        val vt = x._2._2

        val rec = x._2._1.map {
          case Rating(_, product, rating)=>
            "(" + product +","+ rating + ")"
        }.mkString(";")

        vt + "\001" + rec
    }
    println("----Save predictions--------")
    predictions.saveAsTextFile(path="/user/zhangkun6/als/predictions")
    println("----Predictions have been stored!")
  }
}


