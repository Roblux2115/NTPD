from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQL_Parquet") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark dziala poprawnie")
print("Wersja Spark:", spark.version)

data = [
    (1, "Laptop", "Electronics", 3500),
    (2, "Mouse", "Electronics", 80),
    (3, "Desk", "Furniture", 700),
    (4, "Chair", "Furniture", 250)
]

columns = ["id", "product", "category", "price"]

df = spark.createDataFrame(data, columns)

df.write.mode("overwrite").parquet("products_parquet")

print("Zapisano dane do folderu products_parquet")

parquet_df = spark.read.parquet("products_parquet")

print("Dane z pliku Parquet:")
parquet_df.show()

print("Schemat danych:")
parquet_df.printSchema()

spark.stop()