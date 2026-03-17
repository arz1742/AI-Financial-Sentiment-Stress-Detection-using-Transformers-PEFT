import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, roc_auc_score, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def evaluate_inference_speed(func, *args, **kwargs):
    t0     = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def compute_all_metrics(y_true, y_pred, y_prob=None, label_names=None):
    metrics = {
        "accuracy":     accuracy_score(y_true, y_pred),
        "f1_macro":     f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro":     f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred,    average="macro",    zero_division=0),
    }
    if y_prob is not None:
        row_sums = y_prob.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        y_prob_norm = y_prob / row_sums
        n_classes = y_prob_norm.shape[1] if y_prob_norm.ndim == 2 else len(np.unique(y_true))
        try:
            if n_classes > 2:
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_prob_norm, multi_class="ovr", average="macro")
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob_norm[:, 1])
        except ValueError as e:
            logger.warning(f"ROC-AUC skipped: {e}")

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if label_names is not None and len(label_names) == len(per_class_f1):
        for name, score in zip(label_names, per_class_f1):
            metrics[f"f1_{name.lower()}"] = score
    else:
        for i, score in enumerate(per_class_f1):
            metrics[f"f1_class_{i}"] = score

    logger.info("\n" + classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    return metrics


def generate_comparative_report(model_metrics_dict):
    rows = []
    for model_name, m in model_metrics_dict.items():
        row = {"model": model_name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"\n{'='*60}\n  Comparative Evaluation\n{'='*60}\n{df.to_string(index=False)}\n{'='*60}")
    return df
