from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count

spark = SparkSession.builder \
    .appName("DataFrameExample") \
    .getOrCreate()

df = spark.read.csv(
    "sales.csv",
    header=True,
    inferSchema=True
)

print("Dane z pliku CSV")
df.show()

print("Schemat DataFrame")
df.printSchema()

print("Wybrane kolumny: product, category, price")
df.select("product", "category", "price").show()

print("Produkty drozsze niz 500")
df.filter(col("price") > 500).show()

df_with_total = df.withColumn("total_value", col("quantity") * col("price"))

print("Dane z kolumna total_value")
df_with_total.show()

summary = df_with_total.groupBy("category").agg(
    count("*").alias("number_of_orders"),
    sum("total_value").alias("total_sales"),
    avg("price").alias("avg_price")
)

print("Agregacja wedlug kategorii")
summary.show()

summary.toPandas().to_csv(
    "output_sales_summary.csv",
    index=False
)

print("Zapisano wynik do pliku: output_sales_summary.csv")

spark.stop()