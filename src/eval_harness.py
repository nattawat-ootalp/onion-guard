"""One evaluation protocol every candidate method must go through.

WHY THIS EXISTS
Comparing "method A got 0.77" against "method B got 0.81" is meaningless
unless both numbers came out of the same procedure. With 60 heads and 12 per
test fold, the gap between a good method and a bad one is smaller than the
gap between two random fold assignments, so small differences in protocol
(one repeat vs five, threshold at 0.5 vs tuned, features ranked before or
after the split) swamp the effect being measured.

THE PROTOCOL
  * StratifiedGroupKFold(5), grouped on sample_code, repeated `repeats` times
    with different shuffles; every metric is the mean over all folds.
  * A method sees ONLY the training split. It returns a scorer and a decision
    threshold. Anything fitted on the test split — scaling, feature ranking,
    threshold choice, resampling — is leakage and inflates the result.
  * The threshold is part of the method, because accuracy/recall/kappa all
    move with it and the report's bar is stated in those terms. A method that
    wins only at a threshold picked by looking at the test fold has not won.
  * AUC is reported alongside, since it is threshold-free and separates
    "ranks heads better" from "picked a better cut".

USAGE
    from eval_harness import evaluate_method, load_dataset

    def my_method(X_train, y_train, seed):
        model = SomeClassifier(random_state=seed).fit(X_train, y_train)
        return (lambda X: model.predict_proba(X)[:, 1]), 0.5

    result = evaluate_method(my_method, name="my method")
    print(result["summary"])
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, load_config

FEATURES_PATH = ROOT / "data" / "features.csv"
SELECTED_PATH = ROOT / "data" / "selected_features.json"
ID_COL = "sample_code"
LABEL_COL = "compactdry"

METRICS = ["accuracy", "recall", "specificity", "precision", "kappa", "auc"]

# The report's bar, restated here so a comparison table can mark pass/fail
# without every caller re-reading config.json.
BAR = {"accuracy": 0.80, "recall": 0.80, "kappa": 0.61}


def load_dataset(use_selected=True):
    """(df, feature_cols, X, y, groups).

    use_selected=False hands back every column in features.csv — for methods
    whose whole point is to choose their own feature subset inside the fold.
    """
    df = pd.read_csv(FEATURES_PATH)
    available = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    if use_selected and SELECTED_PATH.exists():
        cols = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))["features"]
    else:
        cols = available
    return df, cols, df[cols].values, df[LABEL_COL].values, df[ID_COL].values


def _fold_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "kappa": cohen_kappa_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate_method(fit_fn, name="", X=None, y=None, groups=None,
                    repeats=5, n_splits=5, cfg=None):
    """Score one method.

    fit_fn(X_train, y_train, seed) -> (scorer, threshold)
        scorer(X) must return P(positive) for each row.
        threshold is whatever the method decided, using training data only.

    Returns {"name", "summary" (1-row DataFrame), "folds" (per-fold DataFrame),
             "passes_bar" (dict)}.
    """
    cfg = cfg or load_config()
    if X is None:
        _, _, X, y, groups = load_dataset()

    rows = []
    for repeat in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=cfg["random_seed"] + repeat)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            scorer, threshold = fit_fn(X[tr], y[tr], cfg["random_seed"] + repeat)
            proba = np.asarray(scorer(X[te])).ravel()
            m = _fold_metrics(y[te], proba, threshold)
            m.update({"repeat": repeat, "fold": fold, "threshold": threshold})
            rows.append(m)

    folds = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "name": name,
        **{c: folds[c].mean() for c in METRICS},
        **{f"{c}_sd": folds[c].std() for c in METRICS},
        "threshold_mean": folds["threshold"].mean(),
        "fn_total": int(folds["fn"].sum()),
        "fp_total": int(folds["fp"].sum()),
    }])
    passes = {k: bool(summary[k].iloc[0] >= v) for k, v in BAR.items()}
    return {"name": name, "summary": summary, "folds": folds, "passes_bar": passes}


def compare(results, sort_by="kappa"):
    """Stack several evaluate_method results into one ranked table."""
    table = pd.concat([r["summary"] for r in results], ignore_index=True)
    table = table.sort_values(sort_by, ascending=False).reset_index(drop=True)
    table["passes"] = [
        " ".join(f"{k}✓" if v >= BAR[k] else f"{k}✗" for k, v in
                 (("accuracy", row.accuracy), ("recall", row.recall), ("kappa", row.kappa)))
        for row in table.itertuples()
    ]
    return table


def print_table(table, cols=("name", "accuracy", "recall", "specificity", "kappa", "auc", "passes")):
    show = table[list(cols)].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))


def baseline_rf(X_train, y_train, seed):
    """Current shipped setup: plain RF, threshold 0.5. The number to beat."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    return (lambda X: rf.predict_proba(X)[:, 1]), 0.5


if __name__ == "__main__":
    df, cols, X, y, groups = load_dataset()
    print(f"{len(df)} หัว | {len(cols)} ฟีเจอร์ | "
          f"{int((y == 1).sum())} พบ / {int((y == 0).sum())} ไม่พบ")
    res = evaluate_method(baseline_rf, name="baseline RF (thr 0.5)", X=X, y=y, groups=groups)
    print_table(compare([res]))
