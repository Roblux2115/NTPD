# Laboratorium 01 – Tworzenie modelu ML w Pythonie. Zapisywanie i wersjonowanie modelu

## 1. Przygotowanie środowiska i danych
Utworzono środowisko wirtualne Python oraz zainstalowano biblioteki:
- numpy
- pandas
- scikit-learn
- joblib

Do ćwiczenia wykorzystano zbiór danych Iris dostępny w bibliotece scikit-learn.

W ramach krótkiej analizy danych:
- wyświetlono pierwsze 5 wierszy,
- sprawdzono rozmiar danych,
- sprawdzono typy kolumn.

## 2. Stworzenie modelu ML
Zastosowano algorytm LogisticRegression z biblioteki scikit-learn.

Dane podzielono na:
- 70% zbiór treningowy
- 30% zbiór testowy

Model został wytrenowany i oceniony za pomocą metryki accuracy.

## 3. Zapisanie i ładowanie modelu
Model zapisano do pliku `iris_model_v1.joblib` z użyciem biblioteki `joblib`.

Następnie utworzono osobny skrypt `load_model.py`, w którym wczytano model i wykonano predykcję dla przykładowego rekordu.

## 4. Wersjonowanie modelu w praktyce
Przyjęto proste nazewnictwo wersji modelu:
- `iris_model_v1.joblib`
- `iris_model_v2.joblib`

Wersję modelu należy zwiększyć w przypadku:
- zmiany hiperparametrów,
- poprawy jakości modelu,
- zmiany zbioru danych,
- zmiany cech wejściowych.

Repozytorium może być oznaczone tagiem, np. `v1.0`.

## 5. Różnice między środowiskiem deweloperskim a produkcyjnym
Środowisko deweloperskie służy do testów, eksperymentów i budowy modelu. Środowisko produkcyjne służy do stabilnego działania modelu dla użytkowników końcowych.

Najważniejsze różnice i wyzwania:
- w produkcji ważna jest niezawodność i wydajność,
- trzeba monitorować jakość predykcji,
- dane produkcyjne mogą różnić się od treningowych,
- konieczne jest zarządzanie wersjami modeli i zależności,
- wdrożenia powinny być zautomatyzowane.

Sposoby radzenia sobie z tymi problemami:
- monitorowanie działania modelu,
- okresowy retraining,
- używanie `requirements.txt`,
- stosowanie CI/CD,
- wersjonowanie modeli i kodu.