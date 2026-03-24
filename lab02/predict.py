# predict.py

import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RUN_ID = "9cbfa6b9e4fc49b5926bfbf58c76008e"

wine = load_wine()
X = wine.data
y = wine.target

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

loaded_model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
print("Model wczytany pomyślnie!")

y_pred = loaded_model.predict(X_test)

single_sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(single_sample)
proba = loaded_model.predict_proba(single_sample)

print(f"Przewidywana klasa: {prediction[0]} ({wine.target_names[prediction[0]]})")
print(f"Prawdopodobieństwa: {proba[0]}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")