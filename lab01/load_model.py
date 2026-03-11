from pathlib import Path
import joblib

model_path = Path(__file__).resolve().parent / "iris_model_v1.joblib"
model = joblib.load(model_path)

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Przykładowy rekord:", sample)
print("Przewidziana klasa:", prediction[0])