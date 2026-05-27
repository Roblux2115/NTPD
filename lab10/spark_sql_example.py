from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQLExample") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# wczytanie CSV
df = spark.read.csv(
    "sales_big.csv",
    header=True,
    inferSchema=True
)

print("Dane z pliku CSV:")
df.show()

print("Schemat danych:")
df.printSchema()

# tabela SQL
df.createOrReplaceTempView("products")

# produkty drozsze niz 1000
expensive = spark.sql("""
    SELECT *
    FROM products
    WHERE price > 1000
""")
print("Pierwsze 10 rekordow:")

top10 = spark.sql("""
    SELECT *
    FROM products
    LIMIT 10
""")

top10.show()

print("Produkty drozsze niz 1000:")
expensive.show()

# statystyki kategorii
summary = spark.sql("""
    SELECT
        category,
        COUNT(*) as number_of_products,
        AVG(price) as avg_price,
        MAX(price) as max_price,
        MIN(price) as min_price
    FROM products
    GROUP BY category
""")

print("Podsumowanie kategorii:")
summary.show()

spark.stop()