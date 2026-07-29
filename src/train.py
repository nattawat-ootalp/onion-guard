"""
Phase 3: Random Forest with StratifiedGroupKFold 5-fold CV.

group=sample_code is kept per the report's method even though, in this
project's current shape, each row in features.csv is ALREADY one head (the
4 views were aggregated in Phase 1) — so every group has exactly one row
and StratifiedGroupKFold behaves identically to StratifiedKFold here. It's
kept so the CV code stays correct for free if a future revision ever goes
back to per-view rows.

Per-fold hyperparameter selection uses inner cross-validation on that
fold's training split only (never touching the held-out test split), and
also fits the chosen config with oob_score=True as a second, independent
consistency check on the same training data ("OOB+CV" per the report).

IMPORTANT — how much the numbers below mean depends on the dataset. While
data/mock_metadata.csv exists the rows are generated images, and the metrics
only prove the training+evaluation code runs end-to-end; they say nothing
about real performance. Once real photos replace it that file is gone and
the metrics are a genuine (if small-sample) estimate. Every printout below
states which case it is, so a report figure is never quoted out of context.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix,
)

from common import ROOT, load_config

FEATURES_PATH = ROOT / "data" / "features.csv"
REPORTS_DIR = ROOT / "reports"
# Same signal the web app uses for its "mock data" badge: the generator writes
# this file, real captures never do. One source of truth beats two.
MOCK_MARKER_PATH = ROOT / "data" / "mock_metadata.csv"
# Written by select_features.py. Absent = train on every column in
# features.csv, which is the right default before any selection has been run.
SELECTED_FEATURES_PATH = ROOT / "data" / "selected_features.json"


def resolve_feature_cols(df):
    """Columns to train on, and a line describing where they came from.

    Kept here rather than duplicated in train_final.py so the CV estimate and
    the shipped model can never end up fitted on different feature sets — the
    kind of mismatch that makes a reported accuracy quietly wrong.
    """
    available = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    if not SELECTED_FEATURES_PATH.exists():
        return available, f"ฟีเจอร์ทั้งหมดใน features.csv ({len(available)} ตัว)"

    with open(SELECTED_FEATURES_PATH, encoding="utf-8") as f:
        selected = json.load(f)["features"]
    missing = [c for c in selected if c not in available]
    if missing:
        raise SystemExit(
            f"{SELECTED_FEATURES_PATH.name} ระบุฟีเจอร์ที่ไม่มีใน features.csv: {missing}\n"
            "รัน src/export_features_from_db.py ใหม่ หรือรัน src/select_features.py --write ใหม่"
        )
    return selected, (f"{len(selected)} ตัวจาก {SELECTED_FEATURES_PATH.name} "
                      f"(คัดจาก {len(available)} ตัว)")

LABEL_COL = "compactdry"
ID_COL = "sample_code"

BLUE_SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#0d366b"])
BAR_COLOR = "#2a78d6"
FLAG_COLOR = "#eb6834"
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"


def select_hyperparams(X_train, y_train, groups_train, cfg):
    train_cfg = cfg["train"]
    inner_cv = StratifiedKFold(n_splits=train_cfg["inner_n_splits"], shuffle=True,
                                random_state=cfg["random_seed"])
    best_score, best_params = -np.inf, None
    for params in train_cfg["param_grid"]:
        rf = RandomForestClassifier(random_state=cfg["random_seed"], **params)
        scores = cross_val_score(rf, X_train, y_train, cv=inner_cv, scoring="accuracy")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score


def fold_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def run_cv(df, feature_cols, cfg):
    X = df[feature_cols].values
    y = df[LABEL_COL].values
    groups = df[ID_COL].values
    ids = df[ID_COL].values

    outer_cv = StratifiedGroupKFold(n_splits=cfg["train"]["outer_n_splits"], shuffle=True,
                                     random_state=cfg["random_seed"])

    fold_rows = []
    oof_rows = []
    total_cm = np.zeros((2, 2), dtype=int)
    importances = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        best_params, inner_cv_score = select_hyperparams(X_train, y_train, groups_train, cfg)

        rf = RandomForestClassifier(random_state=cfg["random_seed"], oob_score=True,
                                     bootstrap=True, **best_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        m = fold_metrics(y_test, y_pred)
        m.update({
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_params": best_params,
            "inner_cv_accuracy": inner_cv_score,
            "oob_score": rf.oob_score_,
        })
        fold_rows.append(m)
        total_cm += confusion_matrix(y_test, y_pred, labels=[0, 1])
        importances.append(rf.feature_importances_)

        for sc, yt, yp, proba in zip(ids[test_idx], y_test, y_pred, y_proba):
            oof_rows.append({"sample_code": sc, "fold": fold_i, "y_true": int(yt),
                              "y_pred": int(yp), "y_proba": float(proba)})

        print(f"Fold {fold_i}: train={len(train_idx)} test={len(test_idx)} "
              f"params={best_params} inner_cv_acc={inner_cv_score:.3f} oob={rf.oob_score_:.3f}")
        print(f"         accuracy={m['accuracy']:.3f} recall={m['recall']:.3f} "
              f"specificity={m['specificity']:.3f} precision={m['precision']:.3f} "
              f"f1={m['f1']:.3f} kappa={m['kappa']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    mean_importance = np.mean(importances, axis=0)
    return fold_df, total_cm, mean_importance, oof_df


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    im = ax.imshow(cm, cmap=BLUE_SEQ)
    labels = ["0 (not found)", "1 (found)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("Actual", color=TEXT_PRIMARY, fontsize=10)
    ax.set_title("Combined confusion matrix (summed across 5 folds)", color=TEXT_PRIMARY, fontsize=11)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax * 0.6 else TEXT_PRIMARY
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_oof_roc(oof_df, out_path):
    """ROC of the MODEL itself, built from out-of-fold predicted probabilities
    (each sample scored by a fold model that never trained on it). Distinct
    from Phase 2's baseline ROC, which was a single raw feature."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    y_true = oof_df["y_true"].values
    y_proba = oof_df["y_proba"].values
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=BAR_COLOR, linewidth=2, zorder=3,
            label=f"Random Forest (out-of-fold), AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.set_xlabel("False Positive Rate (1 - specificity)", fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylabel("True Positive Rate (sensitivity)", fontsize=9, color=TEXT_PRIMARY)
    ax.set_title(f"Model ROC — out-of-fold predictions\nAUC = {auc_val:.3f}",
                 fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return auc_val


def plot_feature_importance(feature_cols, mean_importance, watchlist, out_path, top_n=20):
    order = np.argsort(mean_importance)[::-1][:top_n]
    names = [feature_cols[i] for i in order]
    vals = mean_importance[order]
    colors = [FLAG_COLOR if any(name.startswith(w) for w in watchlist) else BAR_COLOR for name in names]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(names))))
    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, vals, color=colors, zorder=3)
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean feature importance (averaged across 5 fold models)", fontsize=9, color=TEXT_PRIMARY)
    ax.set_title(f"Top {top_n} feature importances — Random Forest", fontsize=11, color=TEXT_PRIMARY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if FLAG_COLOR in colors:
        handles = [plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR), plt.Rectangle((0, 0), 1, 1, color=FLAG_COLOR)]
        ax.legend(handles, ["feature", "on the low-importance watchlist (NDFI / A_low)"],
                  loc="lower right", frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = pd.read_csv(FEATURES_PATH)
    feature_cols, feature_source = resolve_feature_cols(df)

    is_mock = MOCK_MARKER_PATH.exists()

    print("=== Phase 3: Random Forest, StratifiedGroupKFold(5) ===")
    if is_mock:
        print("REMINDER: dataset is 100% mock — numbers below prove the pipeline runs, "
              "they are NOT a real accuracy estimate.\n")
    else:
        print(f"Dataset: REAL photos, {len(df)} heads "
              f"({int((df[LABEL_COL] == 1).sum())} พบ / {int((df[LABEL_COL] == 0).sum())} ไม่พบ). "
              f"Features: {feature_source}.")
        print(f"Sample size is small — {len(df) / len(feature_cols):.1f} heads per feature — so "
              "every metric below carries a wide confidence interval. Treat the SD across folds "
              "as the honest error bar, not the mean alone.\n")

    fold_df, total_cm, mean_importance, oof_df = run_cv(df, feature_cols, cfg)
    oof_df.to_csv(REPORTS_DIR / "phase3_oof_predictions.csv", index=False)
    print(f"Wrote {REPORTS_DIR / 'phase3_oof_predictions.csv'}")

    print("\n=== Mean ± SD across folds ===")
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "kappa"]
    summary = fold_df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    print(summary.round(3).to_string())

    fold_df.to_csv(REPORTS_DIR / "phase3_cv_fold_metrics.csv", index=False)
    print(f"\nWrote {REPORTS_DIR / 'phase3_cv_fold_metrics.csv'}")

    plot_confusion_matrix(total_cm, REPORTS_DIR / "phase3_confusion_matrix.png")
    plot_oof_roc(oof_df, REPORTS_DIR / "phase3_model_roc.png")

    imp_df = pd.DataFrame({"feature": feature_cols, "mean_importance": mean_importance})
    imp_df = imp_df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(REPORTS_DIR / "phase3_feature_importance.csv", index=False)
    print(f"Wrote {REPORTS_DIR / 'phase3_feature_importance.csv'}")

    watchlist = cfg["train"]["low_importance_watchlist"]
    plot_feature_importance(feature_cols, mean_importance, watchlist,
                             REPORTS_DIR / "phase3_feature_importance.png")

    print("\n=== Top 15 feature importances ===")
    print(imp_df.head(15).to_string(index=False))

    print("\n=== Baseline comparison (Phase 2) ===")
    blob_related = [f for f in feature_cols if any(
        f.startswith(p) for p in ("n_small_sharp_blobs", "ratio_small_to_large", "avg_small_blob_diam_px", "cluster_density")
    )]
    blob_ranks = imp_df[imp_df["feature"].isin(blob_related)]
    print("Blob-cluster features (the ones Phase 2's baseline was built on):")
    print(blob_ranks.to_string(index=False))
    top_5_names = set(imp_df.head(5)["feature"])
    matches_baseline = bool(set(blob_related) & top_5_names)
    print(f"-> {'YES' if matches_baseline else 'NO'}: a blob-cluster feature is in the top 5 "
          f"importances, {'matching' if matches_baseline else 'NOT matching'} the Phase 2 baseline's signal.")

    print("\n=== NDFI / A_low watchlist ===")
    watch_rows = imp_df[imp_df["feature"].str.startswith(tuple(watchlist))]
    if watch_rows.empty:
        # Feature selection already removed every watchlisted feature, so there
        # is no importance to compare. Without this guard .max() returns NaN,
        # the comparison is False, and the "actually non-trivial" branch prints
        # the exact opposite of what happened.
        print(f"ไม่มีฟีเจอร์ในรายการเฝ้าระวัง ({', '.join(watchlist)}) อยู่ในชุดที่เทรนแล้ว "
              "— ถูกคัดออกไปตอนเลือกฟีเจอร์")
    elif watch_rows["mean_importance"].max() < imp_df["mean_importance"].median():
        print(watch_rows.to_string(index=False))
        print("NOTE: NDFI/A_low importances are below the median feature — consistent with "
              "the Phase 2 univariate AUC finding. Keeping them in the feature list as agreed "
              + ("(not deleting); revisit once real photos are available, since a formula/behavior "
                 "that looks weak on mock data may or may not hold on real fluorescence."
                 if is_mock else
                 "(not deleting). This now holds on REAL fluorescence too, so they are candidates "
                 "for removal — with this few heads per feature, dropping dead weight helps."))
    else:
        print(watch_rows.to_string(index=False))
        print("NOTE: NDFI/A_low actually show non-trivial importance here — worth a closer look "
              "before assuming they're safe to ignore.")

    print("\n=== Report's pass/fail bar ==="
          + (" (NOT meaningful on mock data, checked only for completeness)" if is_mock else ""))
    train_cfg = cfg["train"]
    mean_acc, mean_rec, mean_kappa = summary.loc["accuracy", "mean"], summary.loc["recall", "mean"], summary.loc["kappa", "mean"]
    passed = mean_acc >= train_cfg["accuracy_min"] and mean_rec >= train_cfg["recall_min"] and mean_kappa >= train_cfg["kappa_min"]
    print(f"Accuracy {mean_acc:.3f} (need >= {train_cfg['accuracy_min']}), "
          f"Recall {mean_rec:.3f} (need >= {train_cfg['recall_min']}), "
          f"Kappa {mean_kappa:.3f} (need >= {train_cfg['kappa_min']}) "
          f"-> {'PASS' if passed else 'FAIL'}"
          + (" (meaningless until real data replaces mock)" if is_mock else ""))


if __name__ == "__main__":
    main()
