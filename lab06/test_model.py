import numpy as np
from model import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość listy predykcji jest większa od 0
    i czy odpowiada liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match test labels length."


def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości predykcji mieszczą się w spodziewanym zakresie.
    Dla naszego modelu są 2 klasy: 0 - nie zdał, 1 - zdał.
    """
    preds, _ = train_and_predict()
    assert np.all(np.isin(preds, [0, 1])), "Predictions should contain only 0 or 1."


def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    accuracy = get_accuracy()
    assert accuracy >= 0.7, "Model accuracy should be at least 70%."