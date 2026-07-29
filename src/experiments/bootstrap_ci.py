"""How wide is the uncertainty on the winner's score?

The verification run put GradBoost d2 at kappa 0.620 against a bar of 0.61.
A mean that lands 0.010 above a threshold is not evidence of clearing it —
the question is what range of true values is consistent with 60 heads.

Resampling HEADS (not folds) is the right unit: reshuffling folds only
measures split luck, while the real question is what happens with a different
sample of onions from the same population. Out-of-fold predictions are
generated once, then heads are drawn with replacement and the metric
recomputed on each draw.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_config  # noqa: E402
from eval_harness import BAR, load_dataset  # noqa: E402

N_BOOT = 4000
N_REPEATS = 20


def oof_predictions(make_model, X, y, groups, cfg, repeats=N_REPEATS):
    """Average out-of-fold probability per head across repeated CV.

    Averaging over repeats before bootstrapping keeps the two sources of
    noise separate: this function absorbs split luck, the bootstrap below
    measures sampling of heads.
    """
    acc = np.zeros(len(y))
    for r in range(repeats):
        seed = cfg["random_seed"] + 100 + r
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            m = make_model(seed)
            m.fit(X[tr], y[tr])
            acc[te] += m.predict_proba(X[te])[:, 1]
    return acc / repeats


def boot_ci(y, proba, threshold=0.5, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pred = (proba >= threshold).astype(int)
    kap, accs, specs, recs = [], [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yt, yp = y[idx], pred[idx]
        if len(set(yt)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        kap.append(cohen_kappa_score(yt, yp))
        accs.append(accuracy_score(yt, yp))
        specs.append(tn / (tn + fp) if tn + fp else np.nan)
        recs.append(tp / (tp + fn) if tp + fn else np.nan)
    return {"kappa": np.array(kap), "accuracy": np.array(accs),
            "specificity": np.array(specs), "recall": np.array(recs)}


def main():
    cfg = load_config()
    _, cols, X, y, groups = load_dataset()

    models = {
        "GradBoost d2": lambda s: GradientBoostingClassifier(
            max_depth=2, n_estimators=100, learning_rate=0.1, random_state=s),
        "baseline RF": lambda s: RandomForestClassifier(
            n_estimators=400, random_state=s, n_jobs=-1),
    }

    rows = []
    for name, make in models.items():
        print(f"รัน {name} ...", flush=True)
        proba = oof_predictions(make, X, y, groups, cfg)
        b = boot_ci(y, proba)
        for metric, vals in b.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            bar = BAR.get(metric)
            rows.append({
                "model": name, "metric": metric,
                "point": float(np.mean(vals)), "ci_lo": float(lo), "ci_hi": float(hi),
                "bar": bar,
                # Share of resamples clearing the bar: how often a fresh sample
                # of 60 heads like this one would pass.
                "p_above_bar": float((vals >= bar).mean()) if bar else np.nan,
            })

    out = pd.DataFrame(rows)
    print("\n=== ช่วงความเชื่อมั่น 95% (bootstrap สุ่มหัวหอมใหม่ 4000 ครั้ง) ===")
    print(out.round(3).to_string(index=False))

    path = Path(__file__).resolve().parent.parent.parent / "reports/experiments/bootstrap_ci.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
