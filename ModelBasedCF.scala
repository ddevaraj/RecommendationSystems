import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object ModelBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("ModelBasedCF").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new  SparkContext(conf)

   /**** Load and Parse train data ****/
    val traindata1 = sc.textFile(args(0))
    val header_train = traindata1.first()
    var traindata = traindata1.filter(line => line!= header_train)

    /**** Load and Parse test data ****/
    val testdata1 = sc.textFile(args(1))
    val header_test = testdata1.first()
    val testdata = testdata1.filter(line => line!= header_test)

    var train_user = traindata.map(_.split(",")).map(line => (line(0)))
    var train_business = traindata.map(_.split(",")).map(line => (line(1)))
    var test_user = testdata.map(_.split(",")).map(line => (line(0)))
    var test_business = testdata.map(_.split(",")).map(line => (line(1)))

    train_user = train_user.union(test_user).distinct()
    train_business = train_business.union(test_business).distinct()

    /**** Train the model ****/
    val trainuser = train_user.zipWithIndex.collectAsMap
    val trainbusiness = train_business.zipWithIndex.collectAsMap

    val trainratings = traindata.map(_.split(",") match
    {case Array(user,business,star) => Rating(trainuser(user).toInt, trainbusiness(business).toInt, (star).toDouble)})
    val testratings = testdata.map(_.split(",") match
    {case Array(user,business,star) => Rating(trainuser(user).toInt, trainbusiness(business).toInt, (star).toDouble)})
    var user_avg = testdata.map(line => line.split(",")).map{x => (trainuser(x(0)).toInt,x(2).toDouble)}.groupByKey().map{x=>(x._1, x._2.reduce(_+_)/ x._2.count(x=>true))}
    var  business_avg = testdata.map(line => line.split(",")).map{x => (trainbusiness(x(1)).toInt,x(2).toDouble)}.groupByKey().map{x=>(x._1, x._2.reduce(_+_)/ x._2.count(x=>true))}

    /**** Predict for testdata ****/
    val rank = 20
    val numIterations = 20
    val als = new ALS()
    als.setIterations(20)
    als.setRank(2)
    als.setSeed(300)
    als.setLambda(0.25)
    val model = als.run(trainratings)

    val user_products = testratings.map{ case Rating(user, business, star) => (user, business)}
    val predictions = model.predict(user_products).map{case Rating(user, business, star) => ((user, business),star)}
    val test_prediction = testratings.map{case Rating(user, business, star) => ((user, business), star)}.join(predictions)
    val mse = test_prediction.map{ case((user, product), (r1,r2))=>
    val err = (r1 - r2)
    err * err}.mean()
    val rmse = math.sqrt(mse)

    val accuracy_info = test_prediction.map{ case((user,product),(r1,r2)) => math.abs(r1-r2)}

    /*** Accuracy Information ***/
    var val1, val2, val3,val4, val5 = 0
    for(value <- accuracy_info.collect) {
      if (value >= 0 && value < 1) {
        val1 = val1 + 1
      }
      else if (value >= 1 && value < 2) {
        val2 = val2 + 1
      }
      else if (value >= 2 && value < 3) {
        val3 = val3 + 1
      }
      else if (value >= 3 && value < 4) {
        val4 = val4 + 1
      }
      else if (value >= 4) {
        val5 = val5 + 1
      }
    }
    val end_time = System.currentTimeMillis()

    println(">=0 and <1: "+ val1)
    println(">=1 and <2: "+ val2)
    println(">=2 and <3: "+ val3)
    println(">=3 and <4: "+ val4)
    println(">=4: "+ val5)
    println("RMSE: " + rmse)
    println("Time: "+ (end_time - start_time)/1000+" sec")

    val user_id_map = trainuser.map(pair => pair.swap)
    val business_id_map = trainbusiness.map(pair => pair.swap)
    var final_pred = predictions.map{ case ((user, business), star) => (user_id_map(user), business_id_map(business), star)}.sortBy{ case(x,y,z) => (x,y)}
    final_pred.map(elem=> elem._1+","+elem._2+","+elem._3).coalesce(1, shuffle = false).saveAsTextFile("Deepthi_Devaraj_ModelBasedCF.txt")
  }
}
