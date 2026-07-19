"""
End-to-end runner
=================
    python run_pipeline.py

Does: load features/labels/groups -> grouped train/test split -> train per-drug
calibrated models -> evaluate honestly on held-out GROUPS -> save metrics +
models. Prints the numbers you should put in your demo.

Run `python src/make_synthetic.py` first if you don't have real data yet.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent / "src"))
from train import train_drug, save_models, predict_raw          # noqa: E402
from decision import decide                                      # noqa: E402
from evaluate import core_metrics, call_summary, by_group, risk_coverage_curve  # noqa: E402
from features import curated_determinant_columns                 # noqa: E402
from target_gate import load_gate                                # noqa: E402


def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    p = cfg["paths"]
    feats = pd.read_parquet(p["features"])
    labels = pd.read_csv(p["labels"])
    groups = pd.read_csv(p["groups"]).set_index("genome_id")["group_id"]
    gate = load_gate()
    curated = set(curated_determinant_columns(feats))

    # --- grouped train/test split (held-out GROUPS, never near-identical leaks) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg["seed"])
    idx = np.arange(len(feats))
    tr_idx, te_idx = next(gss.split(idx, groups=groups.loc[feats.index].values))
    train_ids = feats.index[tr_idx]
    test_ids = feats.index[te_idx]
    print(f"Split: {len(train_ids)} train genomes / {len(test_ids)} test genomes "
          f"across {groups.loc[train_ids].nunique()}+{groups.loc[test_ids].nunique()} "
          f"disjoint groups.")

    models, report = {}, {}
    for abx in cfg["antibiotics"]:
        dm = train_drug(
            abx,
            feats.loc[train_ids],
            labels[labels.genome_id.isin(train_ids)],
            groups.loc[train_ids],
            cfg["calibration"]["method"],
            cfg["calibration"]["cv_folds"],
            cfg["seed"],
        )
        if dm is None:
            continue
        models[abx] = dm

        # ---- evaluate on held-out groups ----
        lab = labels[(labels.antibiotic == abx) &
                     (labels.genome_id.isin(test_ids))].set_index("genome_id")
        common = [g for g in test_ids if g in lab.index]
        y = (lab.loc[common, "label"].str.upper() == "R").astype(int).values

        p_res, verdict_labels = [], []
        for gid in common:
            row = feats.loc[gid]
            pr = predict_raw(dm, row)
            p_res.append(pr)
            present = {c for c in feats.columns if row.get(c, 0) == 1}
            v = decide(abx, pr, present, curated, gate,
                       containment_to_train=1.0,  # real data: use sourmash OOD
                       cfg=cfg)
            verdict_labels.append(v.label)
        p_res = np.array(p_res)

        m = core_metrics(y, p_res)
        m.update(call_summary(common, y, p_res, verdict_labels))
        m["by_group"] = by_group(y, p_res, groups.loc[common].values)
        m["risk_coverage"] = risk_coverage_curve(y, p_res)[::max(1, len(y)//10)]
        report[abx] = m
        print(f"\n[{abx}] balanced_acc={m['balanced_accuracy']:.3f}  "
              f"PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.3f}  "
              f"no-call={m['no_call_rate']:.2f}  "
              f"acc_on_calls={m['accuracy_on_calls_made']:.3f}")

    Path("artifacts").mkdir(exist_ok=True)
    save_models(models, Path(p["models"]))
    Path("artifacts/metrics.json").write_text(json.dumps(report, indent=2))
    print("\nSaved models -> artifacts/models, metrics -> artifacts/metrics.json")


if __name__ == "__main__":
    main()
