"""PR-AUC / ROC-AUC evaluation (AnoDDAE metrics)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Higher score = more anomalous. Returns percentages (0–100) to match paper tables."""
    y = np.asarray(y_true).astype(np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch y={y.shape} scores={s.shape}")
    if len(np.unique(y)) < 2:
        return {"PR-AUC": float("nan"), "ROC-AUC": float("nan"), "n": float(len(y))}

    # Guard constant scores
    if np.nanstd(s) < 1e-12:
        s = s + np.linspace(0, 1e-8, num=len(s))

    pr = float(average_precision_score(y, s))
    roc = float(roc_auc_score(y, s))
    return {
        "PR-AUC": 100.0 * pr,
        "ROC-AUC": 100.0 * roc,
        "n": float(len(y)),
        "n_pos": float((y == 1).sum()),
    }


# Paper DDAE Table 1 macros (percent)
PAPER_DDAE = {
    "unsupervised": {"PR-AUC": 32.77, "ROC-AUC": 74.08},
    "semi-supervised": {"PR-AUC": 61.36, "ROC-AUC": 83.17},
}
