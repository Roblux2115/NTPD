# Laboratorium 01 – Model ML i wersjonowanie

## 1. Dane i przygotowanie środowiska

W projekcie wykorzystano zbiór danych **Iris** z biblioteki scikit-learn.

Zainstalowane biblioteki:
- numpy
- pandas
- scikit-learn
- joblib

Wykonano krótką analizę danych:
- wyświetlenie pierwszych wierszy
- sprawdzenie rozmiaru zbioru
- sprawdzenie typów danych

## 2. Stworzenie modelu

Model został zbudowany przy użyciu algorytmu **Logistic Regression**.

Dane podzielono na:
- 70% zbiór treningowy
- 30% zbiór testowy

Model oceniono przy użyciu:
- accuracy
- classification report

## 3. Zapis i wczytanie modelu

Model zapisano do pliku:

iris_model_v1.joblib

Do zapisu użyto biblioteki **joblib**.

W pliku `load_model.py` model jest wczytywany i wykonywana jest przykładowa predykcja.

## 4. Wersjonowanie modelu

Modele można wersjonować np.:

iris_model_v1.joblib  
iris_model_v2.joblib  

Nową wersję modelu tworzy się gdy:
- zmieniono algorytm
- zmieniono hiperparametry
- zmieniono dane treningowe
## 5. Środowisko developerskie vs produkcyjne

Środowisko developerskie służy do eksperymentów i trenowania modelu.

Środowisko produkcyjne służy do używania modelu przez użytkowników i wymaga stabilności oraz monitorowania jakości predykcji.