from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AdvancedSparkSQL") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

sales_df = spark.read.csv(
    "sales_transactions.csv",
    header=True,
    inferSchema=True
)

products_df = spark.read.csv(
    "products.csv",
    header=True,
    inferSchema=True
)

print("Dane sprzedazy:")
sales_df.show()

print("Dane produktow:")
products_df.show()

sales_df.createOrReplaceTempView("sales")
products_df.createOrReplaceTempView("products")

summary = spark.sql("""
    SELECT
        region,
        COUNT(*) as number_of_sales,
        SUM(amount) as total_sales,
        AVG(amount) as avg_sales
    FROM sales
    GROUP BY region
""")

print("Podsumowanie sprzedazy:")
summary.show()


high_sales = spark.sql("""
    SELECT *
    FROM sales
    WHERE amount > 1000
""")

print("Transakcje powyzej 1000:")
high_sales.show()


joined = spark.sql("""
    SELECT
        s.transaction_id,
        p.product_name,
        p.category,
        s.region,
        s.amount
    FROM sales s
    JOIN products p
    ON s.product_id = p.product_id
""")

print("JOIN danych:")
joined.show()

joined.write.mode("overwrite").csv(
    "joined_output",
    header=True
)

print("Zapisano wynik do folderu joined_output")

spark.stop()