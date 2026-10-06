"""Confidence-threshold logic and evaluation metrics."""
import numpy as np


def top_prediction(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_label, confidence) for each text.

    Confidence is the highest class probability from predict_proba.
    """
    proba = model.predict_proba(texts)
    best_idx = proba.argmax(axis=1)
    return model.classes_[best_idx], proba.max(axis=1)


def threshold_metrics(y_true, y_pred, conf_in, conf_oos, threshold: float) -> dict:
    """Metrics for a given rejection threshold.

    A query is 'accepted' if its confidence >= threshold, otherwise treated as OOS.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accepted = conf_in >= threshold

    return {
        "in_scope_accuracy": float(np.mean(accepted & (y_pred == y_true))),
        "false_rejection_rate": float(np.mean(~accepted)),
        "oos_recall": float(np.mean(conf_oos < threshold)),
    }


def find_best_threshold(y_true, y_pred, conf_in, conf_oos) -> dict:
    """Sweep thresholds; pick the one maximizing mean(in_scope_accuracy, oos_recall)."""
    best = None
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        metrics = threshold_metrics(y_true, y_pred, conf_in, conf_oos, float(threshold))
        score = 0.5 * (metrics["in_scope_accuracy"] + metrics["oos_recall"])
        if best is None or score > best["score"]:
            best = {"threshold": float(threshold), "score": float(score), "metrics": metrics}
    return best