from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Lab09") \
    .getOrCreate()

print("Spark działa poprawnie!")

spark.stop()