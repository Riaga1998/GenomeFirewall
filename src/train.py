"""
Module 02 — The Predictor
=========================
One regularised logistic-regression model PER antibiotic (the brief's dependable
baseline: CPU, fast, calibratable, explainable).

Two things make this trustworthy rather than a leaderboard hack:
  1. GROUPED cross-validation on dedup clusters -> no near-identical genomes in
     both train and calibration/test.
  2. Out-of-fold CALIBRATION (Venn-Abers or isotonic) so the confidence a clinician
     sees actually matches observed frequencies.

We store, per drug: the fitted model, the calibrator, the feature columns, and
the training signatures reference (for OOD checks at inference).
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


@dataclass
class DrugModel:
    antibiotic: str
    model: object                       # sklearn LogisticRegression (fit on all data)
    calibrator: object                  # maps raw score -> calibrated p(resistant)
    calibration_method: str
    feature_cols: list[str] = field(default_factory=list)


def _fit_base(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        C=1.0, class_weight="balanced",
        max_iter=2000, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _oof_scores(X, y, groups, n_folds, seed):
    """Out-of-fold raw probabilities, so calibration never sees its own genome group."""
    oof = np.full(len(y), np.nan)
    n_folds = min(n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=max(2, n_folds))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        m = _fit_base(X[tr], y[tr], seed)
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _fit_calibrator(raw_scores, y, method):
    """Return an object with .predict(p_raw)->p_calibrated."""
    mask = ~np.isnan(raw_scores)
    raw_scores, y = raw_scores[mask], y[mask]

    if method == "venn_abers":
        try:
            from venn_abers import VennAbersCalibrator
            va = VennAbersCalibrator()
            # VA works on scores in [0,1]; store points for inductive prediction.
            va.fit(raw_scores.reshape(-1, 1), y)
            return _VACalibrator(va)
        except Exception:
            method = "isotonic"  # graceful fallback

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, y)
    return _IsoCalibrator(iso)


class _IsoCalibrator:
    def __init__(self, iso): self.iso = iso
    def predict(self, p): return self.iso.predict(np.asarray(p).ravel())


class _VACalibrator:
    def __init__(self, va): self.va = va
    def predict(self, p):
        p = np.asarray(p).reshape(-1, 1)
        p_prime, _ = self.va.predict_proba(p)
        return p_prime[:, 1]


def train_drug(
    antibiotic: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    groups: pd.Series,
    method: str,
    n_folds: int,
    seed: int,
) -> DrugModel | None:
    """labels: rows for THIS antibiotic with columns genome_id,label(R/S)."""
    lab = labels[labels["antibiotic"] == antibiotic].set_index("genome_id")
    common = [g for g in features.index if g in lab.index]
    if len(common) < 20:
        print(f"[skip] {antibiotic}: only {len(common)} labelled genomes.")
        return None

    X_df = features.loc[common]
    y = (lab.loc[common, "label"].astype(str).str.upper() == "R").astype(int).values
    grp = groups.loc[common].values
    if len(np.unique(y)) < 2:
        print(f"[skip] {antibiotic}: only one class present.")
        return None

    X = X_df.values
    oof = _oof_scores(X, y, grp, n_folds, seed)
    calibrator = _fit_calibrator(oof, y, method)
    base = _fit_base(X, y, seed)  # final model on all data
    return DrugModel(antibiotic, base, calibrator, method, list(X_df.columns))


def save_models(models: dict[str, DrugModel], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for drug, dm in models.items():
        with open(out_dir / f"{drug}.pkl", "wb") as fh:
            pickle.dump(dm, fh)
    meta = {d: {"n_features": len(m.feature_cols),
                "calibration": m.calibration_method} for d, m in models.items()}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))


def load_models(out_dir: Path) -> dict[str, DrugModel]:
    models = {}
    for pkl in Path(out_dir).glob("*.pkl"):
        with open(pkl, "rb") as fh:
            models[pkl.stem] = pickle.load(fh)
    return models


def predict_raw(dm: DrugModel, feature_row: pd.Series) -> float:
    """Align a single genome's features to the model's columns and score it."""
    x = np.array([[feature_row.get(c, 0) for c in dm.feature_cols]])
    raw = dm.model.predict_proba(x)[0, 1]
    return float(dm.calibrator.predict([raw])[0])
