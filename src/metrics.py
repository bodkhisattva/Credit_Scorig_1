import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def bin_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }