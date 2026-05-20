from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import shutil


DATA_PATH = "/opt/airflow/data/new_data.csv"
MODELS_DIR = "/opt/airflow/models"
PRODUCTION_DIR = "/opt/airflow/models/production"
PRODUCTION_MODEL_PATH = "/opt/airflow/models/production/production_model.pkl"
PRODUCTION_SCORE_PATH = "/opt/airflow/models/production/production_score.txt"


def retrain_and_validate_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PRODUCTION_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    new_model = RandomForestClassifier(random_state=42)
    new_model.fit(X_train, y_train)

    new_predictions = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, new_predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_model_path = f"{MODELS_DIR}/rf_model_{timestamp}.pkl"

    joblib.dump(new_model, archived_model_path)

    print(f"Nowy model zapisano jako wersje archiwalna: {archived_model_path}")
    print(f"Accuracy nowego modelu: {new_accuracy}")

    old_accuracy = 0.0

    if os.path.exists(PRODUCTION_SCORE_PATH):
        with open(PRODUCTION_SCORE_PATH, "r") as file:
            old_accuracy = float(file.read())

        print(f"Accuracy starego modelu produkcyjnego: {old_accuracy}")
    else:
        print("Brak starego modelu produkcyjnego. Nowy model zostanie ustawiony jako produkcyjny.")

    if new_accuracy > old_accuracy:
        shutil.copy(archived_model_path, PRODUCTION_MODEL_PATH)

        with open(PRODUCTION_SCORE_PATH, "w") as file:
            file.write(str(new_accuracy))

        print("Nowy model jest lepszy. Zostal ustawiony jako model produkcyjny.")
        print(f"Model produkcyjny: {PRODUCTION_MODEL_PATH}")
    else:
        print("Nowy model nie jest lepszy. Pozostaje tylko jako model archiwalny.")

    return archived_model_path


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="retrain_model_dag",
    default_args=default_args,
    description="DAG do re-trenowania, walidacji i warunkowej podmiany modelu ML",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "retraining", "validation"],
) as dag:

    retrain_task = PythonOperator(
        task_id="retrain_and_validate_model",
        python_callable=retrain_and_validate_model,
    )

    retrain_task