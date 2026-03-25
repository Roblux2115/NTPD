# LAB04 

Projekt przedstawia API stworzone w FastAPI z modelem LogisticRegression (scikit-learn), który przewiduje czy student zdał egzamin na podstawie liczby godzin nauki i frekwencji. Aplikacja została uruchomiona lokalnie, w Dockerze oraz w Docker Compose z dodatkowym serwisem Redis.

## Technologie
FastAPI, scikit-learn, NumPy, Docker, Docker Compose, Redis

## Uruchomienie

### Lokalnie
uvicorn app:app --reload

### Docker
docker build -t student-api .
docker run -p 8000:8000 student-api

### Docker Compose
docker compose up --build

## Konfiguracja i wymagania 
Wymagania: Python 3.9+, Docker Desktop

## Endpointy
- `/predict` – predykcja (zdał / nie zdał)
- `/info` – informacje o modelu
- `/health` – status API

## Test (cURL)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"hours_of_study\":7,\"attendance\":90}"