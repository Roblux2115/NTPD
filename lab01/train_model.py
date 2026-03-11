from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# 1. Wczytanie danych
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("Pierwsze 5 wierszy:")
print(df.head())
print("\nRozmiar danych:", df.shape)
print("\nTypy kolumn:")
print(df.dtypes)

# 2. Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Trenowanie modelu
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Ewaluacja
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 5. Zapis modelu
joblib.dump(model, "iris_model_v1.joblib")
print("\nModel zapisany do pliku iris_model_v1.joblib")