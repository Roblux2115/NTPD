import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_predict():
    X_train = np.array([
        [1.0, 40.0],
        [2.0, 50.0],
        [3.0, 55.0],
        [6.0, 80.0],
        [7.0, 90.0],
        [8.0, 95.0]
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [2.5, 52.0],
        [7.5, 92.0],
        [10.0, 70.0]
    ])

    y_test = np.array([0, 1, 1])

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return preds, y_test


def get_accuracy():
    preds, y_test = train_and_predict()
    return accuracy_score(y_test, preds)