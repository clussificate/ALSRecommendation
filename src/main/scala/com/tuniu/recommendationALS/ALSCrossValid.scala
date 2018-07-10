package com.tuniu.recommendationALS

import org.apache.log4j.{Level, LogManager, Logger}
import scopt.OptionParser
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.broadcast.Broadcast



/**
  * 交叉验证，调整参数
  * 输入：rank, alpha,iterations,lambda
  * 输出：决策指标查全率、查准率、F1值
  *      排序指标：AUC
  *
  * 任务提交：
export SPARK_HOME=/opt/local/cluster/spark-2.1.2-SNAPSHOT-bin-2.6.0-cdh5.5.0 && sh $SPARK_HOME/bin/spark-submit --driver-memory 6G --master yarn --class com.tuniu.recommendationALS.ALSCrossValid recommendationALS.jar --rank 10 --numIterations 10 --lambda 1.0 --alpha=1.0

  */

object ALSCrossValid {

  // 定义参数
  case class Params(
                     numIterations: Int = 20,
                     lambda: Double = 1.0,
                     rank: Int = 10,
                     alpha: Double = 1.0) extends AbstractParams[Params]


 //  spark system log
  object Logger2 extends Serializable {
    val logger = LogManager.getRootLogger

    lazy val log = this.logger
  }


  def main(args: Array[String]): Unit = {
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

    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }


  def run(params: Params): Unit = {

    val conf = new SparkConf().setAppName("Implicit ALS Cross Valid")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/user/zhangkun6/als/checkpoint")
    val warehouse_location = "/user/zhangkun6/warehouse"
    val spark = SparkSession.builder
      .appName(" Spark SQL Hive integration ")
      .config("spark.sql.warehouse.dir", warehouse_location)
      .enableHiveSupport()
      .getOrCreate()

    Logger.getRootLogger.setLevel(Level.WARN)

    val implicitPrefs = true
    // 表的位置,数据格式：visitor_trace, poi, rating
    val sql = "select * from tmp_zk_als_log_rt"
    val data = spark.sql(sql).rdd.persist(StorageLevel.MEMORY_AND_DISK)
    val numData = data.count()
    println(s"从Hive中获取的样本数：${numData} ")


    // 由于visitor_trace包含字符无法转换为整型，因此使用zipWithIndex() 函数映射
    val vts = data.map( x => x(0)).distinct()
    val vts_with_indices = vts.zipWithIndex().cache()

    println("映射关系为：")
    vts_with_indices.take(3).foreach(println)
    //  把visitor_trace设为key,并且关联起来
    val reorder_data = data.map (x=> (x(0), (x(1), x(2))))
    val joined = reorder_data.join(vts_with_indices)
    data.unpersist()

    // 生成最终数据
    val ratings_row = joined.map(x => (x._2._2, x._2._1._1,x._2._1._2))
    // 将Any类型转换为String
    val ratings_sort= ratings_row.map { x =>
      val fields = Array(x._1.toInt, x._2.toString.toInt, x._3.toString.toDouble)
        Rating(fields(0).toInt, fields(1).toInt, fields(2))
    }.cache()

    // 根据浏览时长rating数据排序并分区
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
    val numUser  = ratings.map(_.user).distinct().count()
    val numItems = ratings.map(_.product).distinct().count()

    println("样本基本信息为：")
    println("样本数："+numRatings)
    println("vt数："+numUser)
    println("poi数："+numItems)

    val allItemIDs = ratings.map(_.product).distinct().collect()
    val bAllItemIDs = sc.broadcast(allItemIDs)


    // 划分训练集，验证集和测试集
    val Array(training,test) = ratings.randomSplit(Array(0.9,0.1))
    training.cache()
    test.cache()
    ratings.unpersist()


    val numTraining = training.count()
    val numTest=test.count()
    println("验证样本基本信息为：")
    println("训练样本数："+numTraining)
    println("测试样本数："+numTest)

    val model = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setAlpha(params.alpha)
      .setImplicitPrefs(true)
      .setProductBlocks(10)
      .setUserBlocks(10)
      .setCheckpointInterval(10)
      .run(training)

    println(s"Result 0f rank = ${params.rank}, " +
      s"Iterations = ${params.numIterations}, Lambda = ${params.lambda}, alpha = ${params.alpha}")

    //决策指标评估precision, recall, F1值
    println("----------------------------决策指标评估-----------------------------------")
    dicisionEvalution(training,test,model)

    // 排序指标AUC
    println("----------------------------排序指标评估-----------------------------------")
    rankEvalution(training, test,model ,bAllItemIDs)


    println("映射关系为：")
    vts_with_indices.take(3).foreach(println)
    vts_with_indices.unpersist()
    training.unpersist()
    test.unpersist()

  }

  def dicisionEvalution(train:RDD[Rating], test:RDD[Rating],model:MatrixFactorizationModel) : Unit = {
    //训练结果评估
    println("--------------测试集评估如下------------")
    val userPois_train = train.map { case Rating(user, product, rating) => (user, product) }
    val trainPreds = model.predict(userPois_train).map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }
    val ratesAndTrainPreds = train.map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }.join(trainPreds)
    print(calConfusionMatrix(ratesAndTrainPreds).toString())

    // 测试结果评估
    println("--------------测试集评估如下------------")
    val userPois_test = test.map { case Rating(user, product, rating) => (user, product) }
    val testPreds = model.predict(userPois_test).map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }
    val ratesAndTestPreds = test.map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }.join(testPreds)
    print(calConfusionMatrix(ratesAndTestPreds).toString())
  }

  case class ConfusionMatrixResult(accuracy: Double, precision: Double, recall: Double,
                                   fallout: Double, sensitivity: Double, specificity: Double, f: Double) {
    override def toString: String = {
      ("\naccuracy    = %02.4f\n" format accuracy) +
        ("precision   = %02.4f\n" format precision) +
        ("recall      = %02.4f\n" format recall) +
        ("fallout     = %02.4f\n" format fallout) +
        ("sensitivity = %02.4f\n" format sensitivity) +
        ("specificity = %02.4f\n" format specificity) +
        ("f           = %02.4f\n" format f)
    }
    def toListString(delimiter: String): String = {
      Logger2.log.warn(s"toListString()...delimiter=$delimiter")
      (" %02.4f" format accuracy) + delimiter +
        ("%02.4f" format precision) + delimiter +
        ("  %02.4f" format recall) + delimiter +
        ("%02.4f" format fallout) + delimiter +
        ("%02.4f" format sensitivity) + delimiter +
        ("%02.4f" format specificity) + delimiter +
        ("%02.4f" format f)
    }
  }

  case class ConfusionMatrix(tp: Double = 0, fp: Double = 0, fn: Double = 0, tn: Double = 0)

  def calConfusionMatrix(data: RDD[((Int, Int), (Double, Double))]): ConfusionMatrixResult = {

    // 阈值为0.0001,，若要precision更高，该阈值应该设大点
    val epison:Double = 0.0001
    val confusionMatrix = data.flatMap {
      case ((user, product), (fact, pred)) if fact > 0 && pred >= epison =>
        Some(ConfusionMatrix(tp = 1))
      case ((user, product), (fact, pred)) if fact > 0 && pred < epison =>
        Some(ConfusionMatrix(fn = 1))
      case ((user, product), (fact, pred)) if fact <= 0 && pred >= epison =>
        Some(ConfusionMatrix(fp = 1))
      case ((user, product), (fact, pred)) if fact <= 0 && pred < epison =>
        Some(ConfusionMatrix(tn = 1))
      case _ ⇒
        Logger2.log.warn(s"Error: confusionMatrix = $ConfusionMatrix")
        None
    }

    val result = confusionMatrix.reduce((sum, row) ⇒ ConfusionMatrix(sum.tp + row.tp, sum.fp + row.fp, sum.fn + row.fn, sum.tn + row.tn))

    Logger2.log.warn(s"result = ${result}\n")
    val p = result.tp + result.fn
    val n = result.fp + result.tn
    Logger2.log.warn(s"confusionMatrix: p=$p, n=$n\n")

    val accuracy = (result.tp + result.tn) / (p + n)

    val precision = result.tp / (result.tp + result.fp)
    val recall = result.tp / p
    val fallout = result.fp / n
    val sensitivity = result.tp / (result.tp + result.fn)
    val specificity = result.tn / (result.fp + result.tn)

    val f = 2 * ((precision * recall) / (precision + recall))
    Logger2.log.warn(s"F1 value: f=$f\n")
    Logger2.log.warn(s"Results is Accuracy:$accuracy,  Precision: $precision, F1 value: $f")
    ConfusionMatrixResult(accuracy, precision, recall, fallout, sensitivity, specificity, f)
  }



  def rankEvalution(training:RDD[Rating], test:RDD[Rating],model:MatrixFactorizationModel,
                    bAllItemIDs: Broadcast[Array[Int]]) : Unit={

    println("--------------训练集评估如下------------")
    val trainAUC = areaUnderCurve(training, bAllItemIDs, model)
    print(s"AUC : $trainAUC")

    // 测试结果评估
    println("--------------测试集评估如下------------")
    val testAUC = areaUnderCurve(test, bAllItemIDs, model)
    print(s"AUC : $testAUC")
  }

  // 计算AUC
  def areaUnderCurve(positiveData: RDD[Rating],
                      bAllItemIDs: Broadcast[Array[Int]],
                      model:MatrixFactorizationModel):Double = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // 认为测试样本是正样本
    val positiveUserProducts = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions = model.predict(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // 负样本从所有的item中随机选择，并且删除用户的正样本item
    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs => {
        // Init an RNG and the item IDs set once for partition
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          // 每个用户正样本数量和负样本数量一样
          // Duplicates are OK
          while (i < allItemIDs.length && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.length))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          // Result is a collection of (user,negative-item) tuples
          negative.map(itemID => (userID, itemID))
        }
      }
    }.flatMap(t => t).cache()
    println(s"negativeUserProducts.count() :${negativeUserProducts.count}")
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = model.predict(negativeUserProducts).groupBy(_.user)
    negativeUserProducts.unpersist()

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }

}


