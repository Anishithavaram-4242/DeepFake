"""
Evaluation metrics for Deepfake Video Detection.
Calculates Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
Matches Table II of the reference paper.
"""

from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_prob: Predicted probability of being Fake (class 1), shape (N,)
        threshold: Decision threshold for classification
        
    Returns:
        Dictionary containing:
            - accuracy: %
            - precision: %
            - recall: %
            - f1_score: %
            - auc_roc: %
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except ValueError:
        auc = 50.0  # In case only one class is present in batch
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1_score": round(f1, 2),
        "auc_roc": round(auc, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
