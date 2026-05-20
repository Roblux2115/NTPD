import os

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RDDExample") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

sc = spark.sparkContext

rdd = sc.textFile("sales.csv")

header = rdd.first()
data = rdd.filter(lambda x: x != header)
rows = data.map(lambda x: x.split(","))

print("Liczba wierszy:", rows.count())

expensive = rows.filter(lambda x: int(x[4]) > 500)
print("Produkty drozsze niz 500:")
print(expensive.collect())

sales = rows.map(lambda x: int(x[3]) * int(x[4]))
total = sales.reduce(lambda a, b: a + b)
print("Suma sprzedazy:", total)

spark.stop()