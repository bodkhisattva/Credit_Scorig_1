import numpy as np
from src.metrics import bin_metrics

def test_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])

    m = bin_metrics(y_true, y_proba, threshold=0.5)

    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_non_trivial():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.4, 0.6, 0.4, 0.6]) 

    m = bin_metrics(y_true, y_proba, threshold=0.5)

    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
