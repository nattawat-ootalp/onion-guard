"""Choose how many features the model should keep, and which.

WHY
60 heads against 24 features is 2.5 heads per feature. A Random Forest will
happily fit that, but each extra feature is one more chance for a split that
works on these 60 heads and nothing else — which is exactly what the first
real-data run showed: specificity 0.56 with an SD of 0.26 across folds.

METHOD (and the trap it avoids)
Ranking features on the whole dataset and then cross-validating the winners
leaks the test fold into the choice, and reliably reports a few points of
accuracy that do not exist. So the ranking is redone inside every training
split: each fold sees only its own 48 heads, picks its own top-k, and is
scored on heads it never saw. The number reported for each k is therefore an
honest estimate of "pick the top k features this way, on data like this".

Repeats shuffle the fold assignment and average, because with 12 heads per
test fold a single split is mostly noise.

The final feature list is chosen by SELECTION FREQUENCY — how often each
feature made the top-k across every fold of every repeat — not by importance
on the full data. A feature that ranks 3rd in one fold and 20th in the next
is unstable and worth less than one that ranks 8th every time.

    python src/select_features.py
    python src/select_features.py --k 8 --write   # freeze a chosen k
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, load_config
from train import ID_COL, LABEL_COL

FEATURES_PATH = ROOT / "data" / "features.csv"
REPORTS_DIR = ROOT / "reports"
SELECTED_PATH = ROOT / "data" / "selected_features.json"


def drop_redundant(df, feature_cols, threshold=0.95):
    """Collapse near-duplicate features before anything else looks at them.

    mean_R/mean_G/mean_B and the sd_* family measure overlapping things; two
    features correlated at 0.98 split the same information (and the same
    importance) between them, which pushes both down the ranking and can hide
    a genuinely useful third feature. Keeps the first of each correlated pair
    in FEATURE_NAMES order so the choice does not depend on the labels.
    """
    corr = df[feature_cols].corr().abs()
    keep, dropped = [], []
    for col in feature_cols:
        clash = next((k for k in keep if corr.loc[col, k] >= threshold), None)
        if clash is None:
            keep.append(col)
        else:
            dropped.append((col, clash, float(corr.loc[col, clash])))
    return keep, dropped


def rank_features_on(X, y, cfg, n_estimators=400):
    """Importance ranking fit on ONE training split only."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=cfg["random_seed"],
                                 min_samples_leaf=1, n_jobs=-1)
    rf.fit(X, y)
    return rf.feature_importances_


def evaluate_k(df, feature_cols, cfg, k, n_repeats, n_splits):
    """Honest CV score for "keep the top k features", plus which ones won."""
    X_all = df[feature_cols].values
    y = df[LABEL_COL].values
    groups = df[ID_COL].values

    fold_rows, picks = [], Counter()
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=cfg["random_seed"] + repeat)
        for train_idx, test_idx in cv.split(X_all, y, groups):
            imp = rank_features_on(X_all[train_idx], y[train_idx], cfg)
            top = np.argsort(imp)[::-1][:k]
            picks.update(feature_cols[i] for i in top)

            rf = RandomForestClassifier(n_estimators=400, random_state=cfg["random_seed"],
                                        min_samples_leaf=1, n_jobs=-1)
            rf.fit(X_all[train_idx][:, top], y[train_idx])
            proba = rf.predict_proba(X_all[test_idx][:, top])[:, 1]
            pred = (proba >= 0.5).astype(int)
            yt = y[test_idx]

            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            fold_rows.append({
                "accuracy": accuracy_score(yt, pred),
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "kappa": cohen_kappa_score(yt, pred),
                "auc": roc_auc_score(yt, proba) if len(set(yt)) > 1 else np.nan,
            })

    return pd.DataFrame(fold_rows), picks


def main():
    ap = argparse.ArgumentParser(description="เลือกจำนวนและชุดฟีเจอร์ที่เหมาะกับข้อมูลจริง")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=None, help="ค่าเริ่มต้น = จำนวนฟีเจอร์ทั้งหมด")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=None,
                    help="ข้ามการค้นหา ใช้ k นี้เลย (ใช้คู่กับ --write)")
    ap.add_argument("--write", action="store_true",
                    help="บันทึกชุดฟีเจอร์ที่เลือกลง data/selected_features.json")
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = pd.read_csv(FEATURES_PATH)
    all_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    n_splits = cfg["train"]["outer_n_splits"]

    print(f"ข้อมูล {len(df)} หัว ({int((df[LABEL_COL] == 1).sum())} พบ / "
          f"{int((df[LABEL_COL] == 0).sum())} ไม่พบ) | ฟีเจอร์ทั้งหมด {len(all_cols)}")

    cols, dropped = drop_redundant(df, all_cols, args.corr_threshold)
    if dropped:
        print(f"\n=== ตัดฟีเจอร์ที่ซ้ำซ้อน (|corr| >= {args.corr_threshold}) {len(dropped)} ตัว ===")
        for col, clash, c in dropped:
            print(f"  {col:28s} ซ้ำกับ {clash:28s} corr={c:.3f}")
    print(f"เหลือฟีเจอร์ที่เข้าคัดเลือก {len(cols)}")

    cols = np.array(cols)
    k_max = args.k_max or len(cols)
    ks = [args.k] if args.k else list(range(args.k_min, min(k_max, len(cols)) + 1))

    print(f"\n=== ผล CV {n_splits}-fold x {args.repeats} รอบ ต่อจำนวนฟีเจอร์ ===")
    print(f"{'k':>3}  {'accuracy':>14}  {'recall':>14}  {'specificity':>14}  {'kappa':>14}  {'AUC':>8}")
    results, picks_by_k = [], {}
    for k in ks:
        folds, picks = evaluate_k(df, cols, cfg, k, args.repeats, n_splits)
        picks_by_k[k] = picks
        m, s = folds.mean(), folds.std()
        results.append({"k": k, **{f"{c}_mean": m[c] for c in folds.columns},
                        **{f"{c}_std": s[c] for c in folds.columns}})
        print(f"{k:>3}  {m['accuracy']:.3f} ± {s['accuracy']:.3f}  "
              f"{m['recall']:.3f} ± {s['recall']:.3f}  "
              f"{m['specificity']:.3f} ± {s['specificity']:.3f}  "
              f"{m['kappa']:.3f} ± {s['kappa']:.3f}  {m['auc']:>8.3f}")

    res_df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "feature_selection_by_k.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    best = res_df.loc[res_df["kappa_mean"].idxmax()]
    best_k = int(best["k"])
    print(f"\nkappa สูงสุดที่ k={best_k} (kappa {best['kappa_mean']:.3f}, "
          f"accuracy {best['accuracy_mean']:.3f}, AUC {best['auc_mean']:.3f})")
    print("หมายเหตุ: k นี้ถูกเลือกโดยดูตาราง CV ข้างบน จึงมองโลกในแง่ดีกว่าความจริงเล็กน้อย "
          "ตัวเลขจริงของ k ที่เลือกต้องวัดซ้ำด้วยชุดข้อมูลใหม่")

    chosen_k = args.k or best_k
    total_folds = args.repeats * n_splits
    freq = picks_by_k[chosen_k].most_common()
    print(f"\n=== ความถี่ที่ถูกเลือกติด top-{chosen_k} (จาก {total_folds} folds) ===")
    for name, n in freq:
        bar = "#" * int(round(20 * n / total_folds))
        print(f"  {name:28s} {n:>3}/{total_folds}  {bar}")

    selected = [str(name) for name, _ in freq[:chosen_k]]  # str(): cols is a numpy array
    print(f"\nชุดฟีเจอร์ที่เลือก ({len(selected)}): {selected}")

    if args.write:
        payload = {
            "features": selected,
            "k": chosen_k,
            "chosen_by": "selection frequency across repeated nested CV folds",
            "dropped_correlated": [{"feature": c, "duplicate_of": d, "corr": v}
                                    for c, d, v in dropped],
            "cv": {"n_splits": n_splits, "repeats": args.repeats,
                   "kappa_mean": float(res_df.loc[res_df["k"] == chosen_k, "kappa_mean"].iloc[0]),
                   "accuracy_mean": float(res_df.loc[res_df["k"] == chosen_k, "accuracy_mean"].iloc[0])},
            "_note": "อ่านโดย train.py/train_final.py ถ้ามีไฟล์นี้ ลบไฟล์ทิ้ง = กลับไปใช้ฟีเจอร์ทั้งหมด",
        }
        SELECTED_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Wrote {SELECTED_PATH}")


if __name__ == "__main__":
    main()
