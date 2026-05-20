# Laboratorium 08 – Apache Airflow i automatyzacja retrainingu modelu

## Zadanie 1 – Konfiguracja Apache Airflow

W ramach zadania skonfigurowano środowisko Apache Airflow z wykorzystaniem Dockera.

Po uruchomieniu środowiska możliwe było zalogowanie się do interfejsu Airflow pod adresem:

```bash
http://localhost:8080
```

Skonfigurowano również folder `dags/`, w którym przechowywane są własne pliki DAG.

---

## Zadanie 2 – Prosty DAG do re-trenowania modelu

Utworzono własny DAG `retrain_model_dag.py`, którego zadaniem jest automatyczne:

- wczytanie nowych danych,
- wytrenowanie modelu RandomForest,
- zapisanie modelu z wersjonowaniem,
- wykonanie walidacji accuracy.

Modele zapisywane są w folderze:

```bash
models/
```

Każda nowa wersja modelu posiada unikalny timestamp, np.:

```bash
rf_model_20260520_195450.pkl
```

DAG został skonfigurowany do automatycznego uruchamiania według harmonogramu.

---

## Zadanie 3 – Walidacja i warunkowa wymiana modelu

Rozszerzono istniejący DAG o mechanizm walidacji i porównywania modeli.

Po wytrenowaniu nowego modelu wykonywane jest:
- obliczenie accuracy,
- porównanie wyniku ze starszym modelem produkcyjnym,
- decyzja o wdrożeniu nowej wersji modelu.

Jeżeli nowy model osiąga lepszy wynik accuracy, zostaje zapisany jako model produkcyjny w folderze:

```bash
models/production/
```

W przeciwnym przypadku model pozostaje jedynie wersją archiwalną.

Przykładowy wynik działania:

```bash
Accuracy nowego modelu: 0.93
Nowy model jest lepszy. Zostal ustawiony jako model produkcyjny.
```