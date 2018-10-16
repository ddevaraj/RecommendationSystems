import org.apache.spark.{SparkConf, SparkContext}
object ItemBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("ItemBasedCF").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    /*** Read train data ***/
    val data1 = sc.textFile(args(0))
    val train_header = data1.first()
    val train_set1 = data1.filter(elem => elem != train_header).map(_.split(",")).map(elem => ((elem(0), elem(1)), elem(2).toDouble))

    /*** Read test data ***/
    val data2 = sc.textFile(args(1))
    val test_header = data2.first()
    val test_set1 = data2.filter(elem => elem != test_header).map(_.split(",")).map(elem => ((elem(0), elem(1)), elem(2).toDouble))

    /*** Create DS like User-Business mapping, Business-Business mapping ***/
    var test_set = test_set1.collect().map(elem => ((elem._1._1,elem._1._2), 0.0)).toMap
    val user_business = sc.parallelize(train_set1.map(row =>(row._1._1, row._1._2)).collect()).groupByKey().collectAsMap()
    val train_user_map = train_set1.map(elem => (elem._1._1,(elem._1._2, elem._2)))
    val join_train_user_map = train_user_map.join(train_user_map)
    val train_set = join_train_user_map.filter(elem => elem._2._1._1 < elem._2._2._1).map(elem => ((elem._2._1._1, elem._2._2._1), elem._1)).groupByKey()
    val filtered_train_set = train_set.filter(elem => elem._2.size >3)
    val user_mapping = train_set1.collectAsMap()

    /*** Find average rating of users, businesses and all businesses ***/
    var user_avg_map = train_set1.map(line=> (line._1._1, (1,line._2) )).reduceByKey { (val1, val2) => (val1._1 + val2._1, val1._2 + val2._2) }
    var user_avg1: org.apache.spark.rdd.RDD[(String, Double)] = user_avg_map.map(line=> (line._1, line._2._2 / line._2._1 ))
    var user_avg = user_avg1.collectAsMap()
    var all_business_avg = train_set1.map(_._2).sum() / train_set1.count()
    var business_avg = train_set1.map(line=> (line._1._2, (1,line._2) )).reduceByKey {(v1, v2) => (v1._1 + v2._1, v1._2 + v2._2) }.map(row=> (row._1, row._2._2 / row._2._1 )).collectAsMap()

    /*** Calculate similarity between businesses - Pearson Correlation ***/
    val item_similarity = filtered_train_set.map(elem =>{
    val item1 = elem._1._1
    val item2 = elem._1._2
    val users = elem._2
    var avg_rating1: Double = 0.0
    var avg_rating2: Double = 0.0
    var num: Double = 0.0
    var den_firstpart: Double = 0.0
    var den_secondpart: Double = 0.0

    users.map(elem => {
      avg_rating1 += user_mapping(elem, item1)
      avg_rating2 += user_mapping(elem, item2)
    })

    avg_rating1 = avg_rating1/users.size
    avg_rating2 = avg_rating2/users.size
    users.map(row => {
      num += (user_mapping((row, item1))-avg_rating1) * (user_mapping((row, item2)) - avg_rating2)
      den_firstpart += math.pow((user_mapping(row, item1)-avg_rating1),2)
      den_secondpart += math.pow((user_mapping(row, item2)-avg_rating2),2)
    })
    den_firstpart = math.sqrt(den_firstpart)
    den_secondpart = math.sqrt(den_secondpart)
    val den = den_firstpart*den_secondpart
    var sim: Double = 0.0
    if(!(num == 0.toDouble || den == 0.toDouble)){
      sim = num/den
    }
    (elem._1, sim)
    }).collectAsMap()

    test_set = test_set.map(line => {
    var user = line._1._1
    val business =line._1._2
    var list_user:Iterable[String]=Iterable[String]()

    if(user_business.contains(user)) {
      list_user = user_business(user)}
    val scoreThreshold = 0.97
    var list_of_neighbours:List[(String, Double)] = List()
    for (elem <- list_user) {
      var sim: Double = 0.0
      if (!item_similarity.contains(business, elem) && !item_similarity.contains(elem,business)) {
        sim = 0.0
      }
      else {
        if (elem < business) {
          sim = item_similarity(elem, business)
        }
        else if (business < elem) {
          sim = item_similarity(business, elem)
        }
      }
      if (sim> scoreThreshold)
      {
        list_of_neighbours = list_of_neighbours:+((elem,sim))
      }
    }

    var sim_items = list_of_neighbours.take(10)

    /*** Predict the values ***/
    var numerator: Double = 0.0
    numerator = sim_items.map(elem => elem._2 * user_mapping(user, elem._1)).sum
    val denominator = sim_items.map(elem => elem._2).sum
    var pred: Double = 0.0
    if (numerator != 0.toDouble && denominator != 0.toDouble) {
      pred = numerator / denominator
    }
    else if(user_avg.contains(user)){
      pred = user_avg(user)
    }
    else if(business_avg.contains(business)){
      pred = business_avg(business)
    }
    else{
      pred = all_business_avg
    }
    ((user, business), pred)
    })

    val output = sc.parallelize(test_set.toSeq)
    val res = output.join(test_set1)

    /*** RMSE Calculation ***/
    val mse = res.map{ case((user, product), (r1,r2))=>
    val err = (r1 - r2)
    err * err}.mean()
    val rmse = math.sqrt(mse)
    val accuracy_info = res.map{ case((user,product),(r1,r2)) => math.abs(r1-r2)}

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
    var final_pred = output.map{ case ((user, business), star) => (user, business, star)}.sortBy{ case(x,y,z) => (x,y)}
    final_pred.map(elem=> elem._1+","+elem._2+","+elem._3).coalesce(1, shuffle = false).saveAsTextFile("Deepthi_Devaraj_ItemBasedCF.txt")
  }
}
