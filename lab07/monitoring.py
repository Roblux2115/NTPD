import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently.presets import ClassificationPreset
from evidently import Dataset, DataDefinition, BinaryClassification
# 1. Dane historyczne / treningowe

X_train, y_train = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])
df_train["target"] = y_train


# 2. Dane produkcyjne / nowsze

X_prod, y_prod = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=999
)

df_prod = pd.DataFrame(X_prod, columns=[f"feature_{i}" for i in range(5)])
df_prod["target"] = y_prod


# 3. Trenowanie modelu

model = RandomForestClassifier(random_state=42)
model.fit(df_train.drop("target", axis=1), df_train["target"])


# 4. Predykcje modelu

df_train["prediction"] = model.predict(df_train.drop("target", axis=1))
df_prod["prediction"] = model.predict(df_prod.drop("target", axis=1))


# 5. Wstępna analiza danych

print("=== Dane historyczne ===")
print(df_train.head())
print("\nLiczba rekordow:", len(df_train))
print("\nTypy danych:")
print(df_train.dtypes)

print("\n=== Dane produkcyjne ===")
print(df_prod.head())
print("\nLiczba rekordow:", len(df_prod))
print("\nTypy danych:")
print(df_prod.dtypes)

print("\nRozklad klas w danych historycznych:")
print(df_train["target"].value_counts())

print("\nRozklad klas w danych produkcyjnych:")
print(df_prod["target"].value_counts())

# 6. Raport Data Drift - Evidently AI

data_drift_report = Report([
    DataDriftPreset()
])

data_drift_eval = data_drift_report.run(
    current_data=df_prod.drop(columns=["prediction"]),
    reference_data=df_train.drop(columns=["prediction"])
)

data_drift_eval.save_html("data_drift_report.html")

print("\nRaport Data Drift zapisano jako: data_drift_report.html")

# 7. Analiza jakości predykcji

train_accuracy = accuracy_score(df_train["target"], df_train["prediction"])
train_precision = precision_score(df_train["target"], df_train["prediction"])
train_recall = recall_score(df_train["target"], df_train["prediction"])
train_f1 = f1_score(df_train["target"], df_train["prediction"])

prod_accuracy = accuracy_score(df_prod["target"], df_prod["prediction"])
prod_precision = precision_score(df_prod["target"], df_prod["prediction"])
prod_recall = recall_score(df_prod["target"], df_prod["prediction"])
prod_f1 = f1_score(df_prod["target"], df_prod["prediction"])

print("\n=== Jakosc modelu na danych historycznych ===")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-score:", train_f1)

print("\n=== Jakosc modelu na danych produkcyjnych ===")
print("Accuracy:", prod_accuracy)
print("Precision:", prod_precision)
print("Recall:", prod_recall)
print("F1-score:", prod_f1)

print("\n=== Porownanie jakosci ===")
print("Spadek accuracy:", train_accuracy - prod_accuracy)
print("Spadek precision:", train_precision - prod_precision)
print("Spadek recall:", train_recall - prod_recall)
print("Spadek F1:", train_f1 - prod_f1)


# 8. Raport jakosci klasyfikacji - Evidently AI

data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_labels="prediction"
        )
    ],
    categorical_columns=["target", "prediction"]
)

reference_dataset = Dataset.from_pandas(
    df_train,
    data_definition=data_definition
)

current_dataset = Dataset.from_pandas(
    df_prod,
    data_definition=data_definition
)

classification_report = Report([
    ClassificationPreset()
])

classification_eval = classification_report.run(
    current_dataset,
    reference_dataset
)

classification_eval.save_html("classification_quality_report.html")

print("\nRaport jakosci klasyfikacji zapisano jako: classification_quality_report.html")