from sklearn.datasets import make_classification
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = y

df.to_csv("data/new_data.csv", index=False)

print("Zapisano plik data/new_data.csv")