"""Experiment: is any classifier other than RandomForest better on this dataset?

Every variant goes through src/eval_harness.py unchanged (StratifiedGroupKFold(5)
x 5 repeats, grouped on sample_code). A variant is a fit_fn(X_train, y_train,
seed) -> (scorer, threshold); everything it fits -- scaler, hyperparameter
search, decision threshold -- is fitted on the training split only, which is why
each scaled model is wrapped in an sklearn Pipeline instead of scaling X up
front.

Run:
    PYTHONPATH=src .venv/bin/python src/experiments/exp_classifiers.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ROOT
from eval_harness import (
    baseline_rf, compare, evaluate_method, load_dataset, print_table,
)

OUT_PATH = ROOT / "reports" / "experiments" / "exp_classifiers.csv"

# 60 heads / 5 folds -> ~48 rows per training split. Inner CV for
# hyperparameter search and threshold picking therefore uses 5 folds of ~10.
INNER_SPLITS = 5


def scaled(estimator):
    """Scaler + estimator as one Pipeline, so the scaler is refitted on each
    training split and cannot see the test fold."""
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def make_fit(build, threshold=0.5):
    """Turn a `build(seed) -> estimator` factory into a harness fit_fn."""

    def fit_fn(X_train, y_train, seed):
        model = build(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)
        return (lambda X: model.predict_proba(X)[:, 1]), threshold

    return fit_fn


def make_fit_tuned_threshold(build, metric="kappa"):
    """Same, but the decision threshold is chosen by inner cross-validation on
    the training split (out-of-fold probabilities only -- the test fold is never
    touched), then the model is refitted on the whole training split."""

    def fit_fn(X_train, y_train, seed):
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            oof = cross_val_predict(
                build(seed), X_train, y_train, cv=inner, method="predict_proba"
            )[:, 1]
            model = build(seed).fit(X_train, y_train)

        best_thr, best_score = 0.5, -np.inf
        for thr in np.linspace(0.20, 0.80, 61):
            pred = (oof >= thr).astype(int)
            score = (
                cohen_kappa_score(y_train, pred)
                if metric == "kappa"
                else (pred == y_train).mean()
            )
            # Tie-break towards 0.5: a threshold far from 0.5 that only ties on
            # ~48 inner rows is almost certainly fitting inner-fold noise.
            if score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and abs(thr - 0.5) < abs(best_thr - 0.5)
            ):
                best_thr, best_score = thr, score
        return (lambda X: model.predict_proba(X)[:, 1]), float(best_thr)

    return fit_fn


def gridsearch(estimator, grid, scoring="roc_auc"):
    """Light hyperparameter tuning by inner CV. GridSearchCV refits on the
    training split it is handed, so it stays inside the outer fold."""

    def build(seed):
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
        return GridSearchCV(
            estimator(seed), grid, scoring=scoring, cv=inner, n_jobs=-1, refit=True
        )

    return build


# --------------------------------------------------------------------------
# Model factories. Each takes a seed and returns an unfitted estimator.
# --------------------------------------------------------------------------

def b_rf(seed):
    return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)


def b_rf_balanced(seed):
    return RandomForestClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1,
        min_samples_leaf=2, class_weight="balanced_subsample",
    )


def b_extratrees(seed):
    return ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1)


def b_extratrees_leaf2(seed):
    return ExtraTreesClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1, min_samples_leaf=2
    )


def b_logreg_l2(seed):
    return scaled(LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                     random_state=seed))


def b_logreg_l2_tuned(seed):
    return gridsearch(
        lambda s: scaled(LogisticRegression(penalty="l2", max_iter=5000, random_state=s)),
        {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
    )(seed)


def b_logreg_l1_tuned(seed):
    return gridsearch(
        lambda s: scaled(LogisticRegression(penalty="l1", solver="liblinear",
                                            max_iter=5000, random_state=s)),
        {"clf__C": [0.03, 0.1, 0.3, 1, 3, 10, 30]},
    )(seed)


def b_logreg_l2_balanced(seed):
    return scaled(LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                     class_weight="balanced", random_state=seed))


def b_svm_linear(seed):
    return scaled(SVC(kernel="linear", C=1.0, probability=True, random_state=seed))


def b_svm_linear_tuned(seed):
    return gridsearch(
        lambda s: scaled(SVC(kernel="linear", probability=True, random_state=s)),
        {"clf__C": [0.03, 0.1, 0.3, 1, 3, 10]},
    )(seed)


def b_svm_rbf(seed):
    return scaled(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                      random_state=seed))


def b_svm_rbf_tuned(seed):
    return gridsearch(
        lambda s: scaled(SVC(kernel="rbf", probability=True, random_state=s)),
        {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.03, 0.1, 0.3]},
    )(seed)


def b_hgb(seed):
    # Shallow and short: 48 training rows cannot support a deep booster.
    return HistGradientBoostingClassifier(
        max_depth=2, max_iter=150, learning_rate=0.05, min_samples_leaf=5,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


def b_gb_stumps(seed):
    return GradientBoostingClassifier(
        max_depth=1, n_estimators=200, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=seed,
    )


def b_gb_depth2(seed):
    return GradientBoostingClassifier(
        max_depth=2, n_estimators=150, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=seed,
    )


def b_gnb(seed):
    return scaled(GaussianNB())


def b_knn_tuned(seed):
    return gridsearch(
        lambda s: scaled(KNeighborsClassifier()),
        {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
         "clf__weights": ["uniform", "distance"]},
    )(seed)


def b_knn5(seed):
    return scaled(KNeighborsClassifier(n_neighbors=5))


def b_vote_lr_rf(seed):
    return VotingClassifier(
        estimators=[("lr", b_logreg_l2(seed)), ("rf", b_rf(seed))],
        voting="soft",
    )


def b_vote_lr_rf_svm(seed):
    return VotingClassifier(
        estimators=[("lr", b_logreg_l2(seed)), ("rf", b_rf(seed)),
                    ("svm", b_svm_rbf(seed))],
        voting="soft",
    )


def b_vote_lr_et_gb(seed):
    return VotingClassifier(
        estimators=[("lr", b_logreg_l2(seed)), ("et", b_extratrees(seed)),
                    ("gb", b_gb_stumps(seed))],
        voting="soft",
    )


# --------------------------------------------------------------------------

# (label, fit_fn). Threshold 0.5 unless the label says otherwise.
VARIANTS = [
    ("baseline RF 400 (thr 0.5)", baseline_rf),
    ("RF balanced leaf2", make_fit(b_rf_balanced)),
    ("ExtraTrees 400", make_fit(b_extratrees)),
    ("ExtraTrees leaf2", make_fit(b_extratrees_leaf2)),
    ("LogReg L2 C=1", make_fit(b_logreg_l2)),
    ("LogReg L2 balanced", make_fit(b_logreg_l2_balanced)),
    ("LogReg L2 tuned C", make_fit(b_logreg_l2_tuned)),
    ("LogReg L1 tuned C", make_fit(b_logreg_l1_tuned)),
    ("SVM linear C=1", make_fit(b_svm_linear)),
    ("SVM linear tuned C", make_fit(b_svm_linear_tuned)),
    ("SVM RBF default", make_fit(b_svm_rbf)),
    ("SVM RBF tuned C/gamma", make_fit(b_svm_rbf_tuned)),
    ("HistGB shallow d2", make_fit(b_hgb)),
    ("GradBoost stumps d1", make_fit(b_gb_stumps)),
    ("GradBoost d2", make_fit(b_gb_depth2)),
    ("GaussianNB", make_fit(b_gnb)),
    ("kNN k=5", make_fit(b_knn5)),
    ("kNN tuned k", make_fit(b_knn_tuned)),
    ("Vote soft LR+RF", make_fit(b_vote_lr_rf)),
    ("Vote soft LR+RF+SVM", make_fit(b_vote_lr_rf_svm)),
    ("Vote soft LR+ET+GB", make_fit(b_vote_lr_et_gb)),
    # Threshold picked by inner CV on the training split only.
    ("RF + inner-CV thr", make_fit_tuned_threshold(b_rf)),
    ("LogReg L2 + inner-CV thr", make_fit_tuned_threshold(b_logreg_l2)),
    ("SVM RBF + inner-CV thr", make_fit_tuned_threshold(b_svm_rbf)),
    ("Vote LR+RF + inner-CV thr", make_fit_tuned_threshold(b_vote_lr_rf)),
    ("Vote LR+RF+SVM + inner-CV thr", make_fit_tuned_threshold(b_vote_lr_rf_svm)),
]


def paired_vs_baseline(results, base_name):
    """Per-fold paired comparison against the baseline.

    Every variant is scored on exactly the same 25 (repeat, fold) splits, so the
    difference within a fold removes the fold-difficulty variance that dominates
    the marginal SD. Reported: mean delta, SD of the delta, a paired t-test and
    the share of folds the variant does not lose.
    """
    from scipy import stats

    base = next(r for r in results if r["name"] == base_name)["folds"]
    key = ["repeat", "fold"]
    rows = []
    for r in results:
        if r["name"] == base_name:
            continue
        m = r["folds"].merge(base, on=key, suffixes=("", "_base"))
        d_kappa = (m["kappa"] - m["kappa_base"]).values
        d_acc = (m["accuracy"] - m["accuracy_base"]).values
        t, p = stats.ttest_rel(m["kappa"], m["kappa_base"])
        rows.append({
            "name": r["name"],
            "d_kappa_mean": d_kappa.mean(),
            "d_kappa_sd": d_kappa.std(ddof=1),
            "d_kappa_t": t,
            "d_kappa_p": p,
            "d_acc_mean": d_acc.mean(),
            "folds_not_worse": float((d_kappa >= -1e-9).mean()),
        })
    return pd.DataFrame(rows).sort_values("d_kappa_mean", ascending=False)


# Variants re-run with more repeats to check the ranking is not an artefact of
# the 5 default shuffles. Picking the top of a 26-row table is itself a
# selection effect, so the leader has to survive a longer run.
ROBUST_NAMES = [
    "baseline RF 400 (thr 0.5)", "GradBoost d2", "GradBoost stumps d1",
    "Vote soft LR+ET+GB", "HistGB shallow d2", "LogReg L2 C=1",
]


def robustness_check(X, y, groups, repeats=15):
    lookup = dict(VARIANTS)
    rows = []
    print(f"\n=== ตรวจความเสถียร: รันซ้ำ {repeats} รอบ (แทน 5) ===")
    for name in ROBUST_NAMES:
        res = evaluate_method(lookup[name], name=name, X=X, y=y, groups=groups,
                              repeats=repeats)
        s = res["summary"].iloc[0]
        # Repeats 0-4 are the shuffles the main table (and therefore the choice
        # of what to put in ROBUST_NAMES) was made on. Repeats 5+ are fresh
        # shuffles, so they are the honest read on the leader.
        fresh = res["folds"][res["folds"]["repeat"] >= 5]
        rows.append({"name": name, f"kappa_r{repeats}": s.kappa,
                     f"accuracy_r{repeats}": s.accuracy, f"auc_r{repeats}": s.auc,
                     "kappa_fresh_repeats": fresh["kappa"].mean(),
                     "accuracy_fresh_repeats": fresh["accuracy"].mean()})
        print(f"  {name:<32s} acc={s.accuracy:.3f} kappa={s.kappa:.3f} "
              f"(SD {s.kappa_sd:.3f}) auc={s.auc:.3f} | "
              f"เฉพาะรอบใหม่ 10 รอบ: acc={fresh['accuracy'].mean():.3f} "
              f"kappa={fresh['kappa'].mean():.3f}")
    return pd.DataFrame(rows)


def main():
    df, cols, X, y, groups = load_dataset()
    print(f"{len(df)} หัว | {len(cols)} ฟีเจอร์ | "
          f"{int((y == 1).sum())} พบ / {int((y == 0).sum())} ไม่พบ")
    print(f"ฟีเจอร์: {', '.join(cols)}\n")

    results = []
    for name, fit_fn in VARIANTS:
        res = evaluate_method(fit_fn, name=name, X=X, y=y, groups=groups)
        results.append(res)
        s = res["summary"].iloc[0]
        print(f"  {name:<32s} acc={s.accuracy:.3f} kappa={s.kappa:.3f} "
              f"auc={s.auc:.3f}")

    table = compare(results, sort_by="kappa")

    print("\n=== เรียงตาม kappa ===")
    print_table(table)

    print("\n=== ค่าเฉลี่ย ± SD ข้ามโฟลด์ (25 โฟลด์) ===")
    sd_view = table[["name", "accuracy", "accuracy_sd", "kappa", "kappa_sd",
                     "auc", "auc_sd", "threshold_mean"]].copy()
    for c in sd_view.columns:
        if sd_view[c].dtype.kind == "f":
            sd_view[c] = sd_view[c].round(3)
    print(sd_view.to_string(index=False))

    base = table[table["name"] == "baseline RF 400 (thr 0.5)"].iloc[0]
    print(f"\nBaseline RF: kappa={base.kappa:.3f} (SD {base.kappa_sd:.3f}), "
          f"acc={base.accuracy:.3f} (SD {base.accuracy_sd:.3f})")
    print("ส่วนต่าง kappa เทียบ baseline (หน่วย = SD ของ baseline):")
    delta = (table["kappa"] - base.kappa) / base.kappa_sd
    for name, d, k in zip(table["name"], delta, table["kappa"]):
        if name == base["name"]:
            continue
        print(f"  {name:<32s} {k - base.kappa:+.3f} kappa = {d:+.2f} SD")

    paired = paired_vs_baseline(results, base_name="baseline RF 400 (thr 0.5)")
    print("\n=== เทียบแบบจับคู่รายโฟลด์กับ baseline (โฟลด์เดียวกัน 25 โฟลด์) ===")
    print("delta_kappa = kappa(variant) - kappa(baseline) ในโฟลด์เดียวกัน")
    pv = paired.copy()
    for c in pv.columns:
        if pv[c].dtype.kind == "f":
            pv[c] = pv[c].round(3)
    print(pv.to_string(index=False))

    robust = robustness_check(X, y, groups)

    table = table.merge(paired, on="name", how="left")
    table = table.merge(robust, on="name", how="left")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nบันทึกตารางเต็มที่ {OUT_PATH}")


if __name__ == "__main__":
    main()
