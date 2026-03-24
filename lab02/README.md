Projekt demonstruje wykorzystanie MLflow do śledzenia eksperymentów machine learning. Klasyfikujemy gatunki wina (Wine dataset) za pomocą Random Forest, logując wszystkie parametry, metryki i modele.
Technologie:

MLflow - platforma do zarządzania cyklem życia ML
scikit-learn - biblioteka do uczenia maszynowego
Wine dataset - zbiór danych do klasyfikacji 3 gatunków wina (178 próbek, 13 cech)

Co robi projekt:

train.py - trenuje model RandomForestClassifier z 4 różnymi konfiguracjami hiperparametrów (n_estimators, max_depth), logując accuracy i f1_score do MLflow
predict.py - wczytuje zapisany model z MLflow po Run ID i wykonuje predykcję na danych testowych