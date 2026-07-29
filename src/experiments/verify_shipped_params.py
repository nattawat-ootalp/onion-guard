"""Which GradientBoosting configuration should actually ship?

train_final.py's inner CV picked {max_depth 2, n_estimators 50, lr 0.2} on
the full data, but the configuration verified on held-out fold shuffles was
{max_depth 2, n_estimators 100, lr 0.1}. Shipping the first while quoting the
second's numbers would make the reported metrics describe a model that was
never deployed.

Inner CV chose on a single fit over all 60 heads, judged by accuracy — a
noisier signal than 20 shuffles judged by kappa. So the tie is broken by
measurement on the same held-out seeds (142-161) used for the finalist
verification, not by which script happened to run last.
"""

import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_config  # noqa: E402
from eval_harness import BAR, compare, evaluate_method, load_dataset, print_table  # noqa: E402

CANDIDATES = {
    "verified d2/100/0.1": {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
    "innerCV d2/50/0.2": {"max_depth": 2, "n_estimators": 50, "learning_rate": 0.2},
    "stumps d1/100/0.1": {"max_depth": 1, "n_estimators": 100, "learning_rate": 0.1},
    "d2/200/0.05": {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05},
    "d3/100/0.1": {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1},
}


def make_fit(params):
    def fit_fn(X_train, y_train, seed):
        m = GradientBoostingClassifier(random_state=seed, **params)
        m.fit(X_train, y_train)
        return (lambda X: m.predict_proba(X)[:, 1]), 0.5
    return fit_fn


def main():
    cfg = load_config()
    holdout = dict(cfg)
    holdout["random_seed"] = cfg["random_seed"] + 100   # seeds never used to select

    _, cols, X, y, groups = load_dataset()
    print(f"{len(y)} หัว | {len(cols)} ฟีเจอร์ | seed ที่ไม่เคยใช้คัดเลือก, 20 รอบ\n")

    results = []
    for name, params in CANDIDATES.items():
        print(f"  รัน {name} ...", flush=True)
        results.append(evaluate_method(make_fit(params), name=name, X=X, y=y,
                                       groups=groups, repeats=20, cfg=holdout))

    table = compare(results)
    print()
    print_table(table)

    out = Path(__file__).resolve().parent.parent.parent / "reports/experiments/shipped_params.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")

    best = table.iloc[0]
    print(f"\nดีที่สุด: {best['name']} (kappa {best.kappa:.3f})")
    print("เกณฑ์: " + " ".join(
        f"{k} {best[k]:.3f}{'✓' if best[k] >= v else '✗'}" for k, v in BAR.items()))


if __name__ == "__main__":
    main()
