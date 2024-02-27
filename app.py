from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, col, when
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, TimestampType, StringType, DecimalType
from pyspark.ml.feature import VectorAssembler, Tokenizer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.streaming import StreamingContext


# Create a SparkSession object
spark = SparkSession.builder.master('yarn').appName("bike_rental_prediction").getOrCreate()

# Define the schema
schema = StructType([
    StructField("datetime", TimestampType(), False),
    StructField("season", IntegerType(), False),
    StructField("holiday", IntegerType(), False),
    StructField("workingday", IntegerType(), False),
    StructField("weather", IntegerType(), False),
    StructField("temp", FloatType(), False),
    StructField("atemp", FloatType(), False),
    StructField("humidity", IntegerType(), False),
    StructField("windspeed", FloatType(), False),
    StructField("casual", IntegerType(), True),
    StructField("registered", IntegerType(), True),
    StructField("count", IntegerType(), False),
])

# Load the train dataset
bike_rental = spark.read.format("csv").option("header", True).schema(schema).load('train.csv')

# Load the test dataset
bike_rental_predict = spark.read.format("csv").option("header", True).schema(schema).load('test.csv')

# Get summary of data and variable types.
bike_rental.describe().show(truncate=False)

# Add datetime columns
bike_rental = bike_rental.withColumn('year', year(bike_rental["datetime"])) \
    .withColumn('month', month(bike_rental["datetime"])) \
    .withColumn('dayofmonth', dayofmonth(bike_rental["datetime"])) \
    .withColumn('hour', hour(bike_rental["datetime"]))

bike_rental_predict = bike_rental_predict.withColumn('year', year(bike_rental_predict["datetime"])) \
    .withColumn('month', month(bike_rental_predict["datetime"])) \
    .withColumn('dayofmonth', dayofmonth(bike_rental_predict["datetime"])) \
    .withColumn('hour', hour(bike_rental_predict["datetime"]))



# Change numerical seasons and weather into season_num and weather_num.

bike_rental = bike_rental.withColumn('season_1', when(col("season") == 1, 1).otherwise(0)) \
    .withColumn('season_2', when(col("season") == 2, 2).otherwise(0)) \
    .withColumn('season_3', when(col("season") == 3, 3).otherwise(0)) \
    .withColumn('season_4', when(col("season") == 4, 4).otherwise(0)) \

bike_rental = bike_rental.withColumn('weather_1', when(col("weather") == 1, 1).otherwise(0)) \
    .withColumn('weather_2', when(col("weather") == 2, 2).otherwise(0)) \
    .withColumn('weather_3', when(col("weather") == 3, 3).otherwise(0)) \
    .withColumn('weather_4', when(col("weather") == 4, 4).otherwise(0)) \

bike_rental_predict = bike_rental_predict.withColumn('season_1', when(col("season") == 1, 1).otherwise(0)) \
    .withColumn('season_2', when(col("season") == 2, 2).otherwise(0)) \
    .withColumn('season_3', when(col("season") == 3, 3).otherwise(0)) \
    .withColumn('season_4', when(col("season") == 4, 4).otherwise(0)) \

bike_rental_predict = bike_rental_predict.withColumn('weather_1', when(col("weather") == 1, 1).otherwise(0)) \
    .withColumn('weather_2', when(col("weather") == 2, 2).otherwise(0)) \
    .withColumn('weather_3', when(col("weather") == 3, 3).otherwise(0)) \
    .withColumn('weather_4', when(col("weather") == 4, 4).otherwise(0)) \

columns = ['season_1', 'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2',
           'weather_3', 'weather_4', 'holiday', 'workingday', "temp", "atemp", "humidity",
           "windspeed", "year", "month", "dayofmonth", "hour", "count"]

# Choose required columns only
bike_rental = bike_rental.select(*columns)
bike_rental_predict = bike_rental_predict.select(*columns)

vector = VectorAssembler(inputCols=columns[:-1], outputCol='features')

gradient_boosted_reg = GBTRegressor(labelCol="count", featuresCol="features", maxIter=50)
pipeline = Pipeline().setStages([vector, gradient_boosted_reg])
gbt_model = pipeline.fit(bike_rental)
predictions = gbt_model.transform(bike_rental_predict)
predictions.select('features', 'count', 'prediction').show()
evaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error is %s" % rmse)
# Root Mean Squared Error is 44.77686867381825

# Save the model
gbt_model.write().overwrite().save('./bike_rental')

sc = spark.sparkContext


columns = ['season_1', 'season_2', 'season_3', 'season_4', 'weather_1', 'weather_2',
           'weather_3', 'weather_4', 'holiday', 'workingday', "temp", "atemp", "humidity",
           "windspeed", "year", "month", "dayofmonth", "hour", "count"]

# Define the schema
schema = StructType([StructField("season_1", IntegerType(), False),
                     StructField("season_2", IntegerType(), False),
                     StructField("season_3", IntegerType(), False),
                     StructField("season_4", IntegerType(), False),
                     StructField("weather_1", IntegerType(), False),
                     StructField("weather_2", IntegerType(), False),
                     StructField("weather_3", IntegerType(), False),
                     StructField("weather_4", IntegerType(), False),
                     StructField("holiday", IntegerType(), False),
                     StructField("workingday", IntegerType(), False),
                     StructField("temp", FloatType(), False),
                     StructField("atemp", FloatType(), False),
                     StructField("humidity", IntegerType(), False),
                     StructField("windspeed", FloatType(), False),
                     StructField("year", IntegerType(), True),
                     StructField("month", IntegerType(), True),
                     StructField("dayofmonth", IntegerType(), False),
                     StructField("hour", IntegerType(), False),
                    ])


def predict(predict_line):
    data = predict_line.map(lambda x: x.split(','))
    data_row = data.map(lambda row: Row(season_1=int(row[0].strip()), season_2=int(row[1].strip()), season_3=int(row[2].strip()), season_4=int(row[3].strip()),
                                        weather_1=int(row[4].strip()), weather_2=int(row[5].strip()), weather_3=int(row[6].strip()), weather_4=int(row[7].strip()),
                                        holiday=int(row[8].strip()), workingday=int(row[9].strip()), temp=float(row[10].strip()), atemp=float(row[11].strip()),
                                        humidity=int(row[12].strip()), windspeed=float(row[13].strip()), year=int(row[14].strip()), month=int(row[15].strip()),
                                        dayofmonth=int(row[16].strip()), hour=int(row[17].strip())))

    df = spark.createDataFrame(data_row, schema)
    df.show()
    loaded_model = PipelineModel.load("./bike_rental")
    streaming_prediction = loaded_model.transform(df)
    streaming_prediction_renamed = streaming_prediction.withColumnRenamed('prediction', 'count').select(*columns)
    streaming_prediction_renamed = streaming_prediction_renamed.withColumn('count', col('count').cast(IntegerType()))
    streaming_prediction_renamed.show()
    streaming_prediction_renamed.write.mode('overwrite').format('jdbc')\
        .option('url',"jdbc:mysql://masked_ip.compute.internal:3306/mased_username")\
        .option('user', 'masked_username').option('password', 'masked_password')\
        .option('driver',"com.mysql.cj.jdbc.Driver")\
        .option('dbtable', 'bike_count').save()

# initialize the streaming context
ssc = StreamingContext(sc, 3)
filepath = "./tmp/kafka/project_topic/project_stream/"
lines = ssc.textFileStream(filepath)
lines.pprint()
# get the predicted sentiments for the tweets received
lines.foreachRDD(predict)

# Start the computation
ssc.start()
ssc.awaitTermination()
