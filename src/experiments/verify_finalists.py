"""Re-test the leading candidates on fold shuffles never used to pick them.

WHY THIS RUN EXISTS
Four experiments scored roughly 100 variants on the same 60 heads, using
repeat seeds 42-46. Taking the best of ~100 on a metric whose fold-to-fold SD
is ~0.25 is a winner's-curse setup: the top of that table is partly real
advantage and partly whichever variant those particular five shuffles
happened to favour. The gap only shows up on data the choice did not touch.

So every finalist is re-run here on seeds 142-161 — twenty fresh shuffles of
the same 60 heads, none of them used during selection. The heads are the
same, so this is not new evidence about onions; it is new evidence about how
much of each variant's lead was fold luck.

What it cannot do: with n=60 the honest uncertainty stays wide no matter how
many times the folds are reshuffled. Read the paired difference against
baseline, not the absolute numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import (  # noqa: E402
    BAR, compare, evaluate_method, load_dataset, print_table,
)

SELECTION_SEEDS = range(5)       # what the experiments used (seed + 0..4)
HOLDOUT_OFFSET = 100             # seeds 142-161, untouched during selection
HOLDOUT_REPEATS = 20

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "reports/experiments/verify_finalists.csv"


# --------------------------------------------------------------- finalists

def baseline_rf(X_train, y_train, seed):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    return (lambda X: rf.predict_proba(X)[:, 1]), 0.5


def gradboost_d2(X_train, y_train, seed):
    """Winner of the classifier sweep: shallow gradient boosting."""
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(max_depth=2, n_estimators=100,
                                   learning_rate=0.1, random_state=seed)
    m.fit(X_train, y_train)
    return (lambda X: m.predict_proba(X)[:, 1]), 0.5


def gradboost_stumps(X_train, y_train, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(max_depth=1, n_estimators=100,
                                   learning_rate=0.1, random_state=seed)
    m.fit(X_train, y_train)
    return (lambda X: m.predict_proba(X)[:, 1]), 0.5


def rf_cw_half(X_train, y_train, seed):
    """Winner of the imbalance sweep: down-weight the majority positive class.

    35:25 makes "positive" the cheap guess; halving its weight prices that in.
    """
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1,
                                class_weight={0: 1.0, 1: 0.5})
    rf.fit(X_train, y_train)
    return (lambda X: rf.predict_proba(X)[:, 1]), 0.5


def rf_cw_balanced(X_train, y_train, seed):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    return (lambda X: rf.predict_proba(X)[:, 1]), 0.5


def gradboost_d2_cw(X_train, y_train, seed):
    """The two independent winners combined, via sample weights.

    GradientBoostingClassifier has no class_weight parameter, so the same
    0.5 down-weighting is applied through sample_weight.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(max_depth=2, n_estimators=100,
                                   learning_rate=0.1, random_state=seed)
    w = np.where(y_train == 1, 0.5, 1.0)
    m.fit(X_train, y_train, sample_weight=w)
    return (lambda X: m.predict_proba(X)[:, 1]), 0.5


def vote_lr_et_gb(X_train, y_train, seed):
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=2000))])
    et = ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    gb = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=seed)
    v = VotingClassifier([("lr", lr), ("et", et), ("gb", gb)], voting="soft")
    v.fit(X_train, y_train)
    return (lambda X: v.predict_proba(X)[:, 1]), 0.5


def l1_infold_rf(X_train, y_train, seed):
    """Winner of the feature-set sweep: L1 picks the columns inside the fold."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train)
    sel = LogisticRegression(penalty="l1", C=0.5, solver="liblinear", max_iter=2000)
    sel.fit(sc.transform(X_train), y_train)
    keep = np.flatnonzero(sel.coef_.ravel() != 0)
    if keep.size == 0:                      # degenerate split: keep everything
        keep = np.arange(X_train.shape[1])
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X_train[:, keep], y_train)
    return (lambda X: rf.predict_proba(X[:, keep])[:, 1]), 0.5


FINALISTS = [
    ("baseline RF (thr 0.5)", baseline_rf),
    ("GradBoost d2", gradboost_d2),
    ("GradBoost stumps d1", gradboost_stumps),
    ("GradBoost d2 + cw 0.5", gradboost_d2_cw),
    ("RF cw pos:neg=0.5:1", rf_cw_half),
    ("RF cw=balanced", rf_cw_balanced),
    ("Vote soft LR+ET+GB", vote_lr_et_gb),
    ("L1 in-fold + RF", l1_infold_rf),
]


def paired_vs_baseline(results, y, X, groups, cfg):
    """Per-fold kappa difference against baseline on the same splits.

    Comparing two means each carrying an SD of 0.25 says almost nothing; the
    same folds are shared by every method, so the paired difference cancels
    the fold-difficulty term that dominates that spread.
    """
    base = next(r for r in results if r["name"] == "baseline RF (thr 0.5)")
    key = ["repeat", "fold"]
    bk = base["folds"].set_index(key)["kappa"]
    rows = []
    for r in results:
        if r["name"] == base["name"]:
            continue
        d = (r["folds"].set_index(key)["kappa"] - bk).dropna()
        sem = d.std() / np.sqrt(len(d))
        rows.append({
            "name": r["name"],
            "d_kappa": d.mean(),
            "sd_of_diff": d.std(),
            "sem": sem,
            # Rough t: |mean| / SEM. Folds within a repeat share training data,
            # so this is optimistic — treat < 2 as "not shown", not as proof.
            "t": abs(d.mean()) / sem if sem > 0 else np.nan,
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("d_kappa", ascending=False)


def main():
    from common import load_config
    cfg = load_config()
    df, cols, X, y, groups = load_dataset()
    print(f"{len(df)} หัว | {len(cols)} ฟีเจอร์ | {int((y==1).sum())} พบ / {int((y==0).sum())} ไม่พบ")
    print(f"ตรวจซ้ำด้วย seed ที่ไม่เคยใช้ตอนคัดเลือก ({HOLDOUT_REPEATS} รอบ)\n")

    holdout_cfg = dict(cfg)
    holdout_cfg["random_seed"] = cfg["random_seed"] + HOLDOUT_OFFSET

    results = []
    for name, fn in FINALISTS:
        print(f"  รัน {name} ...", flush=True)
        results.append(evaluate_method(fn, name=name, X=X, y=y, groups=groups,
                                       repeats=HOLDOUT_REPEATS, cfg=holdout_cfg))

    table = compare(results)
    print("\n=== ผลบน seed ที่ไม่เคยใช้เลือก ===")
    print_table(table)

    print("\n=== ส่วนต่าง kappa เทียบ baseline (จับคู่ fold ต่อ fold) ===")
    paired = paired_vs_baseline(results, y, X, groups, holdout_cfg)
    print(paired.round(3).to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.merge(paired, on="name", how="left").to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    print("\n=== เทียบกับเกณฑ์ในเล่ม ===")
    for row in table.itertuples():
        marks = " ".join(
            f"{k} {getattr(row, k):.3f}{'✓' if getattr(row, k) >= v else '✗'}"
            for k, v in BAR.items())
        print(f"  {row.name:24s} {marks}")


if __name__ == "__main__":
    main()
