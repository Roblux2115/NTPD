Projekt demonstruje stworzenie prostego API z wykorzystaniem FastAPI oraz modelu uczenia maszynowego do predykcji zdania egzaminu. Wykorzystano model LogisticRegression z biblioteki scikit-learn, który na podstawie liczby godzin nauki i frekwencji przewiduje wynik studenta.

Technologie:

FastAPI – framework do budowy API
scikit-learn – biblioteka do ML (LogisticRegression)
NumPy – operacje na danych
Uvicorn – serwer ASGI do uruchamiania aplikacji

Co robi projekt:

app.py – tworzy API oraz trenuje model na przykładowych danych
endpoint /predict – przyjmuje dane (JSON) i zwraca predykcję (zdał / nie zdał)
endpoint /info – zwraca informacje o modelu (typ, cechy)
endpoint /health – sprawdza stan działania API

Aplikacja została przetestowana przy użyciu przeglądarki (/docs) oraz narzędzia cURL i uruchomiona w trybie produkcyjnym przy użyciu Uvicorn.