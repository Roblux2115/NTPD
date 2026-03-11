import joblib

model = joblib.load("iris_model_v1.joblib")

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predykcja dla próbki:", prediction[0])