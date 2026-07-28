"""
Phase 4a: error analysis (from Phase 3's out-of-fold predictions) + train
the final Random Forest on ALL 60 samples + save it for deployment.

Error analysis uses the OOF predictions from train.py's CV (each sample was
predicted exactly once, by a model that never saw it in training) rather
than predictions from the final all-data model, which would trivially
"predict" its own training data. False negatives (compactdry=1 predicted
as 0) get special attention since that's the costly failure mode here —
a real infected head reported as healthy.

Decision threshold: derived from the final model's own out-of-bag
predictions (oob_decision_function_) via Youden's J, the same method
Phase 2 used for the univariate baseline — this avoids picking a cutoff
from in-sample predictions that would look artificially perfect on mock
data.

Saves models/model.joblib (the fitted RandomForestClassifier) and
models/model_config.json (feature order, decision threshold, and the
hyperparameters/seed used) — kept separate from model.joblib itself and
from config.json's image-processing parameters so predict.py's threshold
can be retuned later by editing a small JSON file, no code changes.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

from common import ROOT, load_config
from train import select_hyperparams, LABEL_COL, ID_COL

FEATURES_PATH = ROOT / "data" / "features.csv"
REPORTS_DIR = ROOT / "reports"
OOF_PATH = REPORTS_DIR / "phase3_oof_predictions.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"


def error_analysis():
    print("=== Error analysis (Phase 3 out-of-fold predictions) ===")
    oof_df = pd.read_csv(OOF_PATH)
    wrong = oof_df[oof_df["y_true"] != oof_df["y_pred"]]
    false_neg = wrong[(wrong["y_true"] == 1) & (wrong["y_pred"] == 0)]
    false_pos = wrong[(wrong["y_true"] == 0) & (wrong["y_pred"] == 1)]

    print(f"Misclassified: {len(wrong)}/{len(oof_df)} "
          f"(false negatives: {len(false_neg)}, false positives: {len(false_pos)})")
    if len(false_neg):
        print("\nFalse negatives (compactdry=1 predicted as 0 — infected head called healthy):")
        print(false_neg.to_string(index=False))
    if len(false_pos):
        print("\nFalse positives:")
        print(false_pos.to_string(index=False))
    if len(wrong) == 0:
        print("Nothing to list — 0 misclassifications, consistent with Phase 3's clean CV result. "
              "This is a mock-data artifact (positives are deterministically detectable), NOT a "
              "claim the real classifier will have zero false negatives. Re-run this analysis "
              "once real photos are in — that's when this list will actually mean something.")
    print()


def train_final_model(cfg):
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]

    X = df[feature_cols].values
    y = df[LABEL_COL].values
    groups = df[ID_COL].values

    best_params, inner_cv_score = select_hyperparams(X, y, groups, cfg)
    print(f"=== Final model (trained on all {len(df)} samples) ===")
    print(f"Selected hyperparameters (inner CV on full data): {best_params} (acc={inner_cv_score:.3f})")

    rf = RandomForestClassifier(random_state=cfg["random_seed"], oob_score=True,
                                 bootstrap=True, **best_params)
    rf.fit(X, y)
    print(f"OOB score on full-data fit: {rf.oob_score_:.3f}")

    oob_proba = rf.oob_decision_function_[:, 1]
    fpr, tpr, thresholds = roc_curve(y, oob_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    decision_threshold = float(thresholds[best_idx])
    print(f"Decision threshold (Youden's J on OOB predictions): {decision_threshold:.3f} "
          f"(OOB sensitivity={tpr[best_idx]:.3f}, specificity={1 - fpr[best_idx]:.3f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")

    # Range each feature actually spanned in training. A Random Forest cannot
    # extrapolate: a value far beyond the largest one it ever saw lands in the
    # same leaf as that largest value, so the probability stops responding.
    # Recording the ranges lets predict flag "this input is outside what the
    # model was trained on" instead of returning a confident-looking number.
    ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()),
            "p05": float(df[c].quantile(0.05)), "p95": float(df[c].quantile(0.95))}
        for c in feature_cols
    }

    model_cfg = {
        "_comment": "Model-specific config, separate from config.json's image-processing "
                    "parameters — retune decision_threshold here (e.g. after seeing real "
                    "photos) without touching any code.",
        "feature_names": feature_cols,
        "training_feature_ranges": ranges,
        "_ranges_note": "min/max/p05/p95 of each feature across the training set. Used at "
                        "predict time to detect out-of-distribution input, which a Random "
                        "Forest cannot signal on its own.",
        "decision_threshold": decision_threshold,
        "random_seed": cfg["random_seed"],
        "best_params": best_params,
        "oob_score": rf.oob_score_,
        "trained_on_n_samples": int(len(df)),
        "note": "Trained on 100% MOCK data as of this run — a pipeline proof, not yet fit for "
                "real classification. Retrain once real photos replace data/features.csv.",
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(model_cfg, f, indent=2)
    print(f"Wrote {MODEL_CONFIG_PATH}")


def main():
    cfg = load_config()
    error_analysis()
    train_final_model(cfg)


if __name__ == "__main__":
    main()
