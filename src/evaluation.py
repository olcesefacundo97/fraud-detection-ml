import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-12)

    best_index = int(np.nanargmax(f1_scores[:-1]))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1_score": float(f1_scores[best_index]),
    }


def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold_info = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold_info["threshold"]).astype(int)

        results.append({
            "model": name,
            "threshold": threshold_info["threshold"],
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
        })

    return pd.DataFrame(results)
