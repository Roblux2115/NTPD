
## Zadanie 1 - Parquet

### Uruchomienie

```bash
docker run --rm -v "${PWD}:/app" -w /app spark:python3 /opt/spark/bin/spark-submit parquet_example.py
```

### Wynik

```text
Spark dziala poprawnie
Wersja Spark: 4.1.2

Zapisano dane do folderu products_parquet

Dane z pliku Parquet:

+---+-------+-----------+-----+
| id|product|   category|price|
+---+-------+-----------+-----+
|  1| Laptop|Electronics| 3500|
|  2|  Mouse|Electronics|   80|
|  4|  Chair|  Furniture|  250|
|  3|   Desk|  Furniture|  700|
+---+-------+-----------+-----+

Schemat danych:

root
 |-- id: long (nullable = true)
 |-- product: string (nullable = true)
 |-- category: string (nullable = true)
 |-- price: long (nullable = true)
```

---

## Zadanie 2 - Spark SQL

### Uruchomienie

```bash
docker run --rm -v "${PWD}:/app" -w /app spark:python3 /opt/spark/bin/spark-submit spark_sql_example.py
```

### Wynik

```text
Dane z pliku CSV:
+---+----------+-----------+-----+
| id|   product|   category|price|
+---+----------+-----------+-----+
|  1|    Laptop|Electronics| 3500|
|  2|     Mouse|Electronics|   80|
|  3|      Desk|  Furniture|  700|
|  4|     Chair|  Furniture|  250|
|  5|     Phone|Electronics| 2200|
|  6|  Keyboard|Electronics|  300|
|  7|   Monitor|Electronics| 1200|
|  8|      Sofa|  Furniture| 4000|
|  9|      Lamp|  Furniture|  150|
| 10|    Tablet|Electronics| 1800|
| 11|   Printer|Electronics|  900|
| 12|  Wardrobe|  Furniture| 2700|
| 13|        TV|Electronics| 4500|
| 14|       Bed|  Furniture| 3200|
| 15|Headphones|Electronics|  600|
+---+----------+-----------+-----+

Produkty drozsze niz 1000:
+---+--------+-----------+-----+
| id| product|   category|price|
+---+--------+-----------+-----+
|  1|  Laptop|Electronics| 3500|
|  5|   Phone|Electronics| 2200|
|  7| Monitor|Electronics| 1200|
|  8|    Sofa|  Furniture| 4000|
+---+--------+-----------+-----+

Podsumowanie kategorii:
+-----------+------------------+------------------+---------+---------+
|   category|number_of_products|         avg_price|max_price|min_price|
+-----------+------------------+------------------+---------+---------+
|Electronics|                 9|1675.5555555555557|     4500|       80|
|  Furniture|                 6|1833.3333333333333|     4000|      150|
+-----------+------------------+------------------+---------+---------+
```

---

## Zadanie 3 - Zaawansowane Spark SQL

### Uruchomienie

```bash
docker run --rm -v "${PWD}:/app" -w /app spark:python3 /opt/spark/bin/spark-submit spark_sql_advanced.py
```

### Wynik

```text
Podsumowanie sprzedazy:
+------+---------------+-----------+------------------+
|region|number_of_sales|total_sales|         avg_sales|
+------+---------------+-----------+------------------+
| South|              3|       6000|            2000.0|
|  East|              2|       5700|            2850.0|
|  West|              2|       2700|            1350.0|
| North|              3|       4100|1366.6666666666667|
+------+---------------+-----------+------------------+

Transakcje powyzej 1000:
+--------------+----------+------+------+
|transaction_id|product_id|region|amount|
+--------------+----------+------+------+
|             1|       101| North|  1200|
|             3|       103|  East|  2500|
|             4|       101|  West|  1800|
|             6|       102| South|  1400|
|             7|       105|  East|  3200|
|             9|       104| North|  2200|
|            10|       105| South|  4100|
+--------------+----------+------+------+

JOIN danych:
+--------------+------------+-----------+------+------+
|transaction_id|product_name|   category|region|amount|
+--------------+------------+-----------+------+------+
|             1|      Laptop|Electronics| North|  1200|
|             2|       Mouse|Electronics| South|   500|
|             3|        Desk|  Furniture|  East|  2500|
|             4|      Laptop|Electronics|  West|  1800|
|             6|       Mouse|Electronics| South|  1400|
|             7|       Phone|Electronics|  East|  3200|
|             8|        Desk|  Furniture|  West|   900|
|             9|       Chair|  Furniture| North|  2200|
|            10|       Phone|Electronics| South|  4100|
+--------------+------------+-----------+------+------+

Zapisano wynik do folderu joined_output
```