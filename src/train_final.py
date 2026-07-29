"""
Phase 4a: error analysis (from Phase 3's out-of-fold predictions) + train
the final classifier on ALL 60 samples + save it for deployment.

Error analysis uses the OOF predictions from train.py's CV (each sample was
predicted exactly once, by a model that never saw it in training) rather
than predictions from the final all-data model, which would trivially
"predict" its own training data. False negatives (compactdry=1 predicted
as 0) get special attention since that's the costly failure mode here —
a real infected head reported as healthy.

Decision threshold: derived via Youden's J from out-of-fold probabilities
averaged over repeated cross-validation, the same method Phase 2 used for
the univariate baseline. This was RandomForest's oob_decision_function_
until the model became configurable — GradientBoosting does not bag its
trees and so has no OOB estimate, and either way the point is the same:
never pick a cutoff from rows the model trained on.

Saves models/model.joblib (the fitted classifier, whichever type
config.json's train.model.type names) and
models/model_config.json (feature order, decision threshold, and the
hyperparameters/seed used) — kept separate from model.joblib itself and
from config.json's image-processing parameters so predict.py's threshold
can be retuned later by editing a small JSON file, no code changes.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import ROOT, load_config
from train import (
    make_model, oof_probabilities, resolve_feature_cols, select_hyperparams,
    LABEL_COL, ID_COL,
)

FEATURES_PATH = ROOT / "data" / "features.csv"
REPORTS_DIR = ROOT / "reports"
OOF_PATH = REPORTS_DIR / "phase3_oof_predictions.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
MOCK_MARKER_PATH = ROOT / "data" / "mock_metadata.csv"


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
              + ("This is a mock-data artifact (positives are deterministically detectable), NOT a "
                 "claim the real classifier will have zero false negatives. Re-run this analysis "
                 "once real photos are in — that's when this list will actually mean something."
                 if MOCK_MARKER_PATH.exists() else
                 "On real data with this few heads, a perfect score is more likely to mean the "
                 "split was lucky than that the classifier is perfect."))
    print()


def train_final_model(cfg):
    df = pd.read_csv(FEATURES_PATH)
    feature_cols, feature_source = resolve_feature_cols(df)
    print(f"Features: {feature_source}")

    X = df[feature_cols].values
    y = df[LABEL_COL].values
    groups = df[ID_COL].values

    best_params, inner_cv_score = select_hyperparams(X, y, groups, cfg)
    model_type = cfg["train"].get("model", {}).get("type", "random_forest")
    print(f"=== Final model (trained on all {len(df)} samples) ===")
    print(f"Model type: {model_type}")
    print(f"Selected hyperparameters (inner CV on full data): {best_params} (acc={inner_cv_score:.3f})")

    model = make_model(cfg, params=best_params)
    model.fit(X, y)
    oob = getattr(model, "oob_score_", None)
    if oob is not None:
        print(f"OOB score on full-data fit: {oob:.3f}")

    # Threshold comes from out-of-fold probabilities, never from the fitted
    # model's own training rows — those would look near-perfect and put the
    # cutoff in the wrong place.
    repeats = cfg["train"].get("threshold_cv_repeats", 20)
    print(f"Computing out-of-fold probabilities for the threshold ({repeats} CV repeats)...")
    cv_proba = oof_probabilities(X, y, groups, cfg, best_params, repeats=repeats)

    fpr, tpr, thresholds = roc_curve(y, cv_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    decision_threshold = float(thresholds[best_idx])
    print(f"Decision threshold (Youden's J on out-of-fold predictions): {decision_threshold:.3f} "
          f"(sensitivity={tpr[best_idx]:.3f}, specificity={1 - fpr[best_idx]:.3f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")

    # Range each feature actually spanned in training. No tree ensemble can
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
                        "predict time to detect out-of-distribution input, which a tree "
                        "ensemble cannot signal on its own.",
        "decision_threshold": decision_threshold,
        "_threshold_note": "Youden's J on out-of-fold probabilities averaged over "
                           f"{repeats} CV repeats — never on the model's own training rows.",
        "model_type": model_type,
        "random_seed": cfg["random_seed"],
        "best_params": best_params,
        "oob_score": oob,
        "trained_on_n_samples": int(len(df)),
        "trained_on_mock": MOCK_MARKER_PATH.exists(),
        "note": ("Trained on 100% MOCK data as of this run — a pipeline proof, not yet fit for "
                 "real classification. Retrain once real photos replace data/features.csv."
                 if MOCK_MARKER_PATH.exists() else
                 f"Trained on {len(df)} real heads photographed under 365nm UV, labelled from "
                 "CompactDry YM. Small sample: see reports/phase3_cv_fold_metrics.csv for the "
                 "cross-validated estimate and its spread before quoting any accuracy figure."),
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
