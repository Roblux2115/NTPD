from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np
from fastapi import HTTPException
app = FastAPI(
    title="Student Exam Prediction API",
    description="API przewiduje czy student zdal test na podstawie liczby godzin nauki i frekwencji",
    version="1.0"
)

# Dane treningowe:
# kolumna 1 - liczba godzin nauki
# kolumna 2 - frekwencja (%)
X = np.array([
    [1.0, 40.0],
    [2.0, 50.0],
    [3.0, 55.0],
    [6.0, 80.0],
    [7.0, 90.0],
    [8.0, 95.0]
])

# 0 - nie zdal
# 1 - zdal
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

class InputData(BaseModel):
    hours_of_study: float
    attendance: float

@app.get("/")
def read_root():
    return {"message": "API dziala poprawnie"}

@app.post("/predict")
def predict(data: InputData):

    # Walidacja danych
    if data.hours_of_study < 0:
        raise HTTPException(
            status_code=400,
            detail="Liczba godzin nauki nie moze byc ujemna"
        )

    if data.attendance < 0 or data.attendance > 100:
        raise HTTPException(
            status_code=400,
            detail="Frekwencja musi byc w zakresie 0-100"
        )

    features = np.array([[data.hours_of_study, data.attendance]])
    prediction = model.predict(features)[0]

    result = "zdal" if int(prediction) == 1 else "nie zdal"

    return {
        "prediction": int(prediction),
        "result": result,
        "input_data": {
            "hours_of_study": data.hours_of_study,
            "attendance": data.attendance
        }
    }
@app.get("/info")
def model_info():
    return {
        "model_type": "LogisticRegression",
        "number_of_features": 2,
        "features": ["hours_of_study", "attendance"],
        "description": "Model przewiduje czy student zdal test na podstawie nauki i frekwencji"
    }
@app.get("/health")
def health_check():
    return {"status": "ok"}