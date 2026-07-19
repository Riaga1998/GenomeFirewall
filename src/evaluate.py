"""
Success criteria — honest, not a single headline number
=======================================================
Reports exactly what the brief asks for:
  * balanced accuracy + recall for resistant AND susceptible separately
  * F1, AUROC, PR-AUC per drug
  * Brier score + reliability-curve points
  * no-call rate + accuracy of the calls that ARE made
  * risk-coverage curve (the plot that proves abstention works)
  * breakdown by genetic group (generalisation)

All functions return plain dicts/arrays so you can render them in Streamlit or
matplotlib without this module importing a plotting library.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)


def core_metrics(y_true: np.ndarray, p_resistant: np.ndarray) -> dict:
    """y_true: 1=resistant. p_resistant: calibrated probabilities."""
    y_pred = (p_resistant >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "prevalence_resistant": float(y_true.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_resistant": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_susceptible": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_resistant": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p_resistant)),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, p_resistant))
        out["pr_auc"] = float(average_precision_score(y_true, p_resistant))
    else:
        out["auroc"] = out["pr_auc"] = float("nan")
    return out


def reliability_points(y_true, p, n_bins=10):
    """Calibration curve data: (mean predicted, observed frequency, count) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    pts = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pts.append({"p_mean": float(p[m].mean()),
                    "obs_freq": float(y_true[m].mean()),
                    "count": int(m.sum())})
    return pts


def risk_coverage_curve(y_true, p, abstain_score=None):
    """Selective accuracy vs coverage.

    abstain_score: per-sample confidence used to rank what to keep. Defaults to
    distance from 0.5 (most confident kept first). Returns points of
    (coverage, selective_accuracy).
    """
    if abstain_score is None:
        abstain_score = np.abs(p - 0.5)
    order = np.argsort(-abstain_score)  # most confident first
    y_sorted = y_true[order]
    pred_sorted = (p[order] >= 0.5).astype(int)
    correct = (y_sorted == pred_sorted).astype(int)
    n = len(y_true)
    pts = []
    for k in range(1, n + 1):
        pts.append({"coverage": k / n,
                    "selective_accuracy": float(correct[:k].mean())})
    return pts


def call_summary(labels: list[str], y_true, p_resistant, verdict_labels):
    """How often we abstain, and accuracy of the calls we DO make.

    verdict_labels: the final "likely to fail/work/no-call" per sample.
    """
    made = np.array([v != "no-call" for v in verdict_labels])
    pred = (p_resistant >= 0.5).astype(int)
    acc_made = float((pred[made] == y_true[made]).mean()) if made.any() else float("nan")
    return {
        "no_call_rate": float((~made).mean()),
        "coverage": float(made.mean()),
        "accuracy_on_calls_made": acc_made,
    }


def by_group(y_true, p_resistant, groups):
    """Per-genetic-group metrics — the generalisation view the brief wants."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 5 or len(np.unique(y_true[m])) < 2:
            continue
        out[str(g)] = core_metrics(y_true[m], p_resistant[m])
    return out
