import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    y, preds = [1, 1, 0], [1, 1, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    pred = inference(model, X)

    assert y.shape == pred.shape
