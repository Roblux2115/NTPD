Wynik monitoring.py

=== Dane historyczne ===
   feature_0  feature_1  feature_2  feature_3  feature_4  target  prediction
0  -1.830633  -0.095340  -0.654076   0.724051  -0.181319       0           0
1   0.260281   0.080151  -0.413465  -1.273314   1.482609       1           1
2  -1.379618   0.098744  -0.971657  -0.072798  -1.579555       1           1
3  -0.998061  -0.161506   1.051948   2.398537   2.120715       1           1
4  -0.369610   1.223565   0.621572   0.012779  -1.422353       1           1

Liczba rekordow: 500

Typy danych:
feature_0     float64
feature_1     float64
feature_2     float64
feature_3     float64
feature_4     float64
target          int64
prediction      int64
dtype: object

=== Dane produkcyjne ===
   feature_0  feature_1  feature_2  feature_3  feature_4  target  prediction
0  -3.142705  -1.191936   3.364808   0.312979  -0.323881       1           0
1   0.479566   2.643843   1.069905  -1.026354  -1.693222       1           1
2  -1.009647   0.854427   0.787981   1.085944  -0.776312       0           1
3   0.656041  -0.684570   0.827312  -0.476341  -0.270564       0           0
4  -0.244241   0.982726   1.387011   1.444994   2.486599       0           1

Liczba rekordow: 300

Typy danych:
feature_0     float64
feature_1     float64
feature_2     float64
feature_3     float64
feature_4     float64
target          int64
prediction      int64
dtype: object

Rozklad klas w danych historycznych:
target
1    252
0    248
Name: count, dtype: int64

Rozklad klas w danych produkcyjnych:
target
1    151
0    149
Name: count, dtype: int64

Raport Data Drift zapisano jako: data_drift_report.html

=== Jakosc modelu na danych historycznych ===
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1-score: 1.0

=== Jakosc modelu na danych produkcyjnych ===
Accuracy: 0.42333333333333334
Precision: 0.4266666666666667
Recall: 0.423841059602649
F1-score: 0.42524916943521596

=== Porownanie jakosci ===
Spadek accuracy: 0.5766666666666667
Spadek precision: 0.5733333333333333
Spadek recall: 0.576158940397351
Spadek F1: 0.574750830564784

Raport jakosci klasyfikacji zapisano jako: classification_quality_report.html

## 📊 Model Monitoring (Evidently)

### Dane
- Historyczne: 500 rekordów  
- Produkcyjne: 300 rekordów  
- Dane są zbalansowane (klasy 0/1 ~50/50)

---

### 📉 Jakość modelu

**Dane historyczne:**
- Accuracy: 1.0  
- Precision: 1.0  
- Recall: 1.0  
- F1: 1.0  

**Dane produkcyjne:**
- Accuracy: 0.423  
- Precision: 0.427  
- Recall: 0.424  
- F1: 0.425  

---

### 🔻 Spadek jakości

- Accuracy ↓ ~0.58  
- Precision ↓ ~0.57  
- Recall ↓ ~0.58  
- F1 ↓ ~0.57  

Model działa idealnie na treningu, ale słabo w produkcji.

---

### 📊 Raporty

- `data_drift_report.html` → analiza zmian danych  
- `classification_quality_report.html` → jakość modelu  

---

### 🧠 Wnioski

- Model jest przeuczony (overfitting)  
- Wystąpił data drift / concept drift  
- Model nie generalizuje na nowych danych  

---

### 🔧 Co zrobić

- Retraining modelu  
- Zebrać nowe dane  
- Monitorować model (Evidently)  

---

**Wniosek:**  
Model wymaga ponownego trenowania, ponieważ jego jakość znacząco spadła w produkcji.