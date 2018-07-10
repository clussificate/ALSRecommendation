package com.tuniu.recommendationALS

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


/**
  * 为所有的用户产生TOP10推荐,
  * 用于String型visitor_trace需要映射到Int型
  * 因此该模块仅测试使用，正式场景使用ALSRecommendation.TopNRecommendation
  */

object ALSModelTopN {

  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName(s"ALS Implicit Prediction")
    val sc = new SparkContext(conf)


    val warehouse_location = "/user/zhangkun6/warehouse"

    val spark = SparkSession.builder.appName(" Spark SQL Hive integration ")
      .config("spark.sql.warehouse.dir", warehouse_location)
      .enableHiveSupport().getOrCreate()


    // 表的位置,数据格式：visitor_trace, poi, rating
    val sql = "select * from tmp_zk_als_log_rt_dense"
    val data = spark.sql(sql).rdd

    // 由于visitor_trace包含字符无法转换为整型，因此使用zipWithIndex() 函数映射
    val vts = data.map( x => x(0)).distinct()
    val vts_with_indices = vts.zipWithIndex()
    val indices_with_vts = vts_with_indices.map(x => (x._2.toInt, x._1.toString))



    val model = MatrixFactorizationModel.load(sc,path="/user/zhangkun6/warehouse/als_scala")
    println("-------Model has been loaded!--------")

    // 选择一个用户
//    val user = 5

//    //给用户5推荐评分前10的物品
//    val recommendations = model.recommendProducts(user,10)
//    recommendations.map(x => {
//      println(x.user + "-->" + x.product + "-->" + x.rating)
//    })

    // 为所有的用户产生推荐,推荐结果已经排过序了
    val recommendationForAll = model.recommendProductsForUsers(num=10)
//    val recommendationForALLSorted = recommendationForAll.map{ x =>
//      val u = x._1
//      val rec = x._2.sortBy(x => x.rating)
//      (u, rec)
//    }
    val recForVt = recommendationForAll.join(indices_with_vts)
    /*
      * 解析recForVt的rdd：
      * 原RDD[(Int, (Array[org.apache.spark.mllib.recommendation.Rating], String))]
      * ===>[(uid,(RecArr(Rating), visitor_trace)]
      * 解析为RDD[(String, Array[org.apache.spark.mllib.recommendation.Rating])]
      * ====>[(visitor_trace, ((,);(,)))]
    */

    val predictions = recForVt.map{ x =>
        val vt = x._2._2

        val rec = x._2._1.map {
        case Rating(_, product, rating)=>
            "(" + product.toString +","+ rating.toString + ")"
      }.mkString(";")

        vt + "\001" + rec
    }
    println("----Save predictions--------")
    predictions.saveAsTextFile(path="/user/zhangkun6/als/predictions_test")
  }

}
