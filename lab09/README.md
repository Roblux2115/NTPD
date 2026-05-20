# Laboratorium 09 – Apache Spark i PySpark

---

## Zadanie 1: Uruchomienie lokalnej instancji Apache Spark

Zainstalowano pakiet `pyspark` przy użyciu pip:

```bash
pip install pyspark
```

Poprawność instalacji zweryfikowano prostym skryptem testowym:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Lab09") \
    .getOrCreate()

print("Spark działa poprawnie!")

spark.stop()
```

Skrypt uruchomiono komendą:

```bash
python spark_test.py
```

Wynik:

```
Spark działa poprawnie!
```

**Problem z uruchomieniem zadań 2 i 3 na Windows:**

Podczas próby uruchomienia skryptów `dataframe_operations.py` oraz `rdd_operations.py` na systemie Windows napotkano błąd `Python worker exited unexpectedly (crashed)`. Problem wynikał z konfliktu między zainstalowaną wersją Javy (Java 23) a PySpark, który oficjalnie wspiera Javę 8, 11 lub 17. Dodatkowym utrudnieniem była wersja Python 3.13, która wprowadza zmiany w obsłudze socketów niekompatybilne z mechanizmem komunikacji między JVM a Python workerem w PySpark.

W celu obejścia problemu zadania 2 i 3 uruchomiono wewnątrz kontenera Docker z Pythonem 3.11 i Javą 21:

```bash
docker run --rm -v "${PWD}:/app" -w /app python:3.11 bash -c \
  "apt-get update -q && apt-get install -y default-jdk -q && \
   pip install pyspark==3.5.3 -q && python dataframe_operations.py"
```

```bash
docker run --rm -v "${PWD}:/app" -w /app python:3.11 bash -c \
  "apt-get update -q && apt-get install -y default-jdk -q && \
   pip install pyspark==3.5.3 -q && python rdd_operations.py"
```

---

## Zadanie 2: Podstawowe operacje na DataFrame w PySpark

**Plik danych:** `sales.csv` – dane o sprzedaży produktów elektronicznych i mebli.

**Kod:**

```python
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
```

### Wczytane dane

```
+--------+--------+-----------+--------+-----+------+
|order_id| product|   category|quantity|price|  city|
+--------+--------+-----------+--------+-----+------+
|       1|  Laptop|Electronics|       2| 3500|Warsaw|
|       2|   Mouse|Electronics|       5|   80|Krakow|
|       3|    Desk|  Furniture|       1|  700|Warsaw|
|       4|   Chair|  Furniture|       4|  250|Gdansk|
|       5|Keyboard|Electronics|       3|  150|Krakow|
|       6| Monitor|Electronics|       2|  900|Warsaw|
|       7|    Lamp|  Furniture|       6|  120|Gdansk|
|       8|   Phone|Electronics|       1| 2800|Warsaw|
|       9|    Sofa|  Furniture|       1| 2200|Krakow|
|      10|  Tablet|Electronics|       2| 1600|Gdansk|
+--------+--------+-----------+--------+-----+------+
```

### Schemat DataFrame

```
root
 |-- order_id: integer (nullable = true)
 |-- product: string (nullable = true)
 |-- category: string (nullable = true)
 |-- quantity: integer (nullable = true)
 |-- price: integer (nullable = true)
 |-- city: string (nullable = true)
```

### Wybrane kolumny: product, category, price

```
+--------+-----------+-----+
| product|   category|price|
+--------+-----------+-----+
|  Laptop|Electronics| 3500|
|   Mouse|Electronics|   80|
|    Desk|  Furniture|  700|
|   Chair|  Furniture|  250|
|Keyboard|Electronics|  150|
| Monitor|Electronics|  900|
|    Lamp|  Furniture|  120|
|   Phone|Electronics| 2800|
|    Sofa|  Furniture| 2200|
|  Tablet|Electronics| 1600|
+--------+-----------+-----+
```

### Produkty droższe niż 500

```
+--------+-------+-----------+--------+-----+------+
|order_id|product|   category|quantity|price|  city|
+--------+-------+-----------+--------+-----+------+
|       1| Laptop|Electronics|       2| 3500|Warsaw|
|       3|   Desk|  Furniture|       1|  700|Warsaw|
|       6|Monitor|Electronics|       2|  900|Warsaw|
|       8|  Phone|Electronics|       1| 2800|Warsaw|
|       9|   Sofa|  Furniture|       1| 2200|Krakow|
|      10| Tablet|Electronics|       2| 1600|Gdansk|
+--------+-------+-----------+--------+-----+------+
```

### Dane z kolumną total_value (quantity × price)

```
+--------+--------+-----------+--------+-----+------+-----------+
|order_id| product|   category|quantity|price|  city|total_value|
+--------+--------+-----------+--------+-----+------+-----------+
|       1|  Laptop|Electronics|       2| 3500|Warsaw|       7000|
|       2|   Mouse|Electronics|       5|   80|Krakow|        400|
|       3|    Desk|  Furniture|       1|  700|Warsaw|        700|
|       4|   Chair|  Furniture|       4|  250|Gdansk|       1000|
|       5|Keyboard|Electronics|       3|  150|Krakow|        450|
|       6| Monitor|Electronics|       2|  900|Warsaw|       1800|
|       7|    Lamp|  Furniture|       6|  120|Gdansk|        720|
|       8|   Phone|Electronics|       1| 2800|Warsaw|       2800|
|       9|    Sofa|  Furniture|       1| 2200|Krakow|       2200|
|      10|  Tablet|Electronics|       2| 1600|Gdansk|       3200|
+--------+--------+-----------+--------+-----+------+-----------+
```

### Agregacja według kategorii

```
+-----------+----------------+-----------+---------+
|   category|number_of_orders|total_sales|avg_price|
+-----------+----------------+-----------+---------+
|Electronics|               6|      15650|   1505.0|
|  Furniture|               4|       4620|    817.5|
+-----------+----------------+-----------+---------+
```

Wynik zapisano do pliku: `output_sales_summary.csv`

---

## Zadanie 3: Praca z RDD w PySpark

Dane wczytano jako RDD i ręcznie parsowano wiersze CSV.

**Kod:**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RDDExample") \
    .master("local[1]") \
    .getOrCreate()

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
```

### Liczba wierszy

```
Liczba wierszy: 10
```

### Produkty droższe niż 500 (filter)

```
[['1', 'Laptop', 'Electronics', '2', '3500', 'Warsaw'],
 ['3', 'Desk', 'Furniture', '1', '700', 'Warsaw'],
 ['6', 'Monitor', 'Electronics', '2', '900', 'Warsaw'],
 ['8', 'Phone', 'Electronics', '1', '2800', 'Warsaw'],
 ['9', 'Sofa', 'Furniture', '1', '2200', 'Krakow'],
 ['10', 'Tablet', 'Electronics', '2', '1600', 'Gdansk']]
```

### Suma sprzedaży (map + reduce)

```
Suma sprzedazy: 20270
```

Suma obliczona jako `quantity × price` dla każdego zamówienia, zsumowana przez `reduce`.