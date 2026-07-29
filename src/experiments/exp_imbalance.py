"""Experiment: does class-imbalance handling and/or in-fold threshold selection
beat the plain-RF baseline on the shallot UV dataset?

WHY
The shipped model over-predicts positive (specificity 0.66) which drags kappa
under the report's 0.61 bar. Two families of fix are plausible:
  * change what the learner optimises  -> class_weight / resampling
  * change where the cut is made       -> threshold chosen inside the fold
Both are tested through src/eval_harness.py so every number comes out of the
same 5x5 grouped-CV protocol. Nothing here reads the test fold: every weight,
resample and threshold is derived from X_train/y_train only.

RUN
    PYTHONPATH=src .venv/bin/python src/experiments/exp_imbalance.py [--repeats N] [--jobs N]

Variants are evaluated in parallel processes (one variant per worker, single
threaded fits inside — on 48x10 data a 400-tree fit is 0.21s single-threaded
and 0.88s with n_jobs=-1, so thread fan-out is pure loss here). Every variant's
summary row is appended to the CSV as soon as it finishes, so a killed run
still leaves the variants that completed.
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from common import ROOT
from eval_harness import (
    METRICS, baseline_rf, compare, evaluate_method, load_dataset, print_table,
)

N_TREES = 400
N_JOBS = 1          # 48-row fits; process fan-out costs more than it saves
INNER_SPLITS = 5
INNER_REPEATS = 2   # inner-CV OOF averaged over 2 shuffles to steady the curve
N_UNDERSAMPLE = 15  # members in the undersampling ensemble
UNDERSAMPLE_TREES = 150  # 15 x 150 = 2250 trees total, vs the baseline's 400
OUT_CSV = ROOT / "reports" / "experiments" / "exp_imbalance.csv"


# --------------------------------------------------------------------------
# threshold pickers -- all take (y_train, proba_train) and never see the test fold
# --------------------------------------------------------------------------
def _youden(y, pred):
    pos = y == 1
    neg = ~pos
    tpr = pred[pos].mean() if pos.any() else 0.0
    fpr = pred[neg].mean() if neg.any() else 0.0
    return tpr - fpr


def _kappa(y, pred):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y, pred)


_OBJECTIVES = {"youden": _youden, "kappa": _kappa}


def pick_threshold(y, proba, objective="youden"):
    """Threshold maximising `objective` on (y, proba).

    Candidates are midpoints between observed scores, so every distinct
    labelling of the training rows is considered exactly once. Ties are broken
    by taking the MEDIAN of the winning thresholds: on 48 rows an objective
    plateau is common and its edges sit right next to a training point, which
    generalises worse than its middle.
    """
    score_fn = _OBJECTIVES[objective]
    u = np.unique(proba)
    if u.size < 2:
        return 0.5
    cands = (u[:-1] + u[1:]) / 2.0
    best, winners = -np.inf, []
    for t in cands:
        s = score_fn(y, (proba >= t).astype(int))
        if s > best + 1e-12:
            best, winners = s, [t]
        elif s > best - 1e-12:
            winners.append(t)
    return float(np.median(winners))


# --------------------------------------------------------------------------
# training-fold probability sources for threshold selection
# --------------------------------------------------------------------------
def oob_proba(rf, y_train):
    """Out-of-bag P(positive) for the training rows.

    Rows that never fell out of bag come back NaN; with 400 trees that is
    vanishingly rare, but fall back to the training prevalence so a NaN cannot
    silently become 0.0 and drag the chosen threshold down.
    """
    p = np.asarray(rf.oob_decision_function_)[:, 1].astype(float)
    bad = ~np.isfinite(p)
    if bad.any():
        p[bad] = float(y_train.mean())
    return p


def inner_oof_proba(X, y, seed, model_fn):
    """Out-of-fold P(positive) from an inner CV run on the training fold only."""
    acc = np.zeros(len(y), dtype=float)
    for rep in range(INNER_REPEATS):
        skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                              random_state=seed + 1000 * rep)
        for tr, va in skf.split(X, y):
            m = model_fn(seed + 1000 * rep)
            m.fit(X[tr], y[tr])
            acc[va] += m.predict_proba(X[va])[:, 1]
    return acc / INNER_REPEATS


# --------------------------------------------------------------------------
# model builders
# --------------------------------------------------------------------------
def rf_builder(class_weight=None, oob=False, n_trees=N_TREES):
    def build(seed):
        return RandomForestClassifier(
            n_estimators=n_trees, random_state=seed, class_weight=class_weight,
            oob_score=oob, bootstrap=True, n_jobs=N_JOBS,
        )
    return build


def calibrated_builder(method, class_weight=None, cv=5):
    def build(seed):
        return CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=N_TREES, random_state=seed,
                class_weight=class_weight, n_jobs=N_JOBS),
            method=method, cv=cv,
        )
    return build


class UndersampleEnsemble:
    """Bagged random undersampling of the majority class.

    n_members RFs, each fitted on a balanced draw (all minority rows + an equal
    number of majority rows sampled without replacement), predictions averaged.
    n_members=1 is plain single-shot undersampling. The draw uses only the
    training labels handed to fit().
    """

    def __init__(self, seed, n_members=N_UNDERSAMPLE, class_weight=None,
                 n_trees=N_TREES):
        self.seed = seed
        self.n_members = n_members
        self.class_weight = class_weight
        self.n_trees = n_trees

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n = min(len(pos), len(neg))
        self.models_ = []
        for k in range(self.n_members):
            idx = np.concatenate([
                rng.choice(pos, n, replace=False) if len(pos) > n else pos,
                rng.choice(neg, n, replace=False) if len(neg) > n else neg,
            ])
            rf = RandomForestClassifier(
                n_estimators=self.n_trees, random_state=self.seed + k,
                class_weight=self.class_weight, n_jobs=N_JOBS)
            rf.fit(X[idx], y[idx])
            self.models_.append(rf)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])


def undersample_builder(n_members=N_UNDERSAMPLE, class_weight=None,
                        n_trees=UNDERSAMPLE_TREES):
    def build(seed):
        return UndersampleEnsemble(seed, n_members=n_members,
                                   class_weight=class_weight, n_trees=n_trees)
    return build


# --------------------------------------------------------------------------
# method factory: model builder + threshold rule -> fit_fn for the harness
# --------------------------------------------------------------------------
def make_method(build, thr_mode):
    """thr_mode:
        ("fixed", v)          -- constant cut, no tuning
        ("oob", objective)    -- maximise objective on the RF's OOB predictions
        ("inner", objective)  -- maximise objective on inner-CV OOF predictions
        ("insample", obj)     -- maximise objective on resubstitution predictions
                                 (NOT leakage of the test fold, but optimistic
                                 on the training fold; kept as a control)
    """
    kind, arg = thr_mode

    def fit_fn(X_train, y_train, seed):
        model = build(seed)
        model.fit(X_train, y_train)
        if kind == "fixed":
            thr = float(arg)
        elif kind == "oob":
            thr = pick_threshold(y_train, oob_proba(model, y_train), arg)
        elif kind == "inner":
            thr = pick_threshold(y_train, inner_oof_proba(X_train, y_train, seed, build), arg)
        elif kind == "insample":
            thr = pick_threshold(y_train, model.predict_proba(X_train)[:, 1], arg)
        else:
            raise ValueError(kind)
        return (lambda X: model.predict_proba(X)[:, 1]), thr

    return fit_fn


def build_variants():
    """(name, fit_fn) for every variant, in the order they are evaluated."""
    v = [("baseline RF thr0.5", baseline_rf)]

    # 1. built-in class weighting
    for cw, label in [("balanced", "balanced"), ("balanced_subsample", "bal_subsample")]:
        v.append((f"RF cw={label} thr0.5", make_method(rf_builder(cw), ("fixed", 0.5))))

    # 2. explicit weight dicts. r < 1 down-weights the positive class, i.e. the
    #    direction that should buy specificity back. r > 1 included as a control.
    for r in [0.4, 0.5, 0.6, 0.714, 0.8, 1.4]:
        v.append((f"RF cw pos:neg={r:g}:1 thr0.5",
                  make_method(rf_builder({0: 1.0, 1: r}), ("fixed", 0.5))))

    # 3. moving the cut without any tuning at all, as a reference for how much
    #    of any threshold-tuning gain is just "cut higher than 0.5"
    for t in [0.55, 0.6, 0.65]:
        v.append((f"RF thr{t:g} (fixed)", make_method(rf_builder(), ("fixed", t))))

    # 4. threshold chosen inside the training fold
    for obj in ["youden", "kappa"]:
        v.append((f"RF thr@OOB-{obj}", make_method(rf_builder(oob=True), ("oob", obj))))
        v.append((f"RF thr@innerCV-{obj}", make_method(rf_builder(), ("inner", obj))))
    v.append(("RF thr@insample-kappa (control)",
              make_method(rf_builder(), ("insample", "kappa"))))

    # 5. calibration, with and without threshold tuning
    for m in ["sigmoid", "isotonic"]:
        v.append((f"RF+calib-{m} thr0.5", make_method(calibrated_builder(m), ("fixed", 0.5))))
        v.append((f"RF+calib-{m} thr@innerCV-kappa",
                  make_method(calibrated_builder(m), ("inner", "kappa"))))

    # 6. random undersampling of the majority (positive) class
    v.append(("RF undersample x1 thr0.5",
              make_method(undersample_builder(1, n_trees=N_TREES), ("fixed", 0.5))))
    v.append((f"RF undersample x{N_UNDERSAMPLE} thr0.5",
              make_method(undersample_builder(), ("fixed", 0.5))))
    v.append((f"RF undersample x{N_UNDERSAMPLE} thr@innerCV-kappa",
              make_method(undersample_builder(), ("inner", "kappa"))))

    # 7. combinations
    v.append(("RF cw=balanced thr@OOB-kappa",
              make_method(rf_builder("balanced", oob=True), ("oob", "kappa"))))
    v.append(("RF cw=balanced thr@innerCV-kappa",
              make_method(rf_builder("balanced"), ("inner", "kappa"))))
    v.append(("RF cw pos:neg=0.6:1 thr@OOB-kappa",
              make_method(rf_builder({0: 1.0, 1: 0.6}, oob=True), ("oob", "kappa"))))
    v.append(("RF cw pos:neg=0.6:1 thr@innerCV-kappa",
              make_method(rf_builder({0: 1.0, 1: 0.6}), ("inner", "kappa"))))
    v.append(("RF cw pos:neg=0.6:1 thr@OOB-youden",
              make_method(rf_builder({0: 1.0, 1: 0.6}, oob=True), ("oob", "youden"))))
    v.append(("RF cw pos:neg=0.5:1 thr@innerCV-kappa",
              make_method(rf_builder({0: 1.0, 1: 0.5}), ("inner", "kappa"))))
    v.append((f"RF undersample x{N_UNDERSAMPLE} thr0.6",
              make_method(undersample_builder(), ("fixed", 0.6))))
    v.append(("RF cw=bal_subsample thr@OOB-kappa",
              make_method(rf_builder("balanced_subsample", oob=True), ("oob", "kappa"))))
    # (No "OOB cut from the plain RF applied to calibrated scores" variant:
    #  a threshold only means something on the score scale it was fitted on.)
    return v


def paired_deltas(results, ref_name):
    """Fold-by-fold difference against the reference method.

    The harness drives every method through the same CV seeds, so fold i of
    repeat r is literally the same 12 heads for all variants. Pairing removes
    the fold-to-fold variance that dominates the raw SD column, and is the only
    honest way to tell a real gain from a lucky split at n=60.
    """
    ref = {r["name"]: r for r in results}[ref_name]["folds"]
    rows = []
    for r in results:
        d = r["folds"]["kappa"].values - ref["kappa"].values
        da = r["folds"]["accuracy"].values - ref["accuracy"].values
        rows.append({
            "name": r["name"],
            "d_kappa_mean": d.mean(), "d_kappa_sd": d.std(ddof=1),
            "d_kappa_sem": d.std(ddof=1) / np.sqrt(len(d)),
            "d_acc_mean": da.mean(),
            "win_folds": int((d > 1e-9).sum()), "lose_folds": int((d < -1e-9).sum()),
        })
    return pd.DataFrame(rows)


def _run_one(index, X, y, groups, repeats):
    """Evaluate variant `index`. The variant list is rebuilt inside the worker
    so no closure has to cross the process boundary."""
    name, fn = build_variants()[index]
    return evaluate_method(fn, name=name, X=X, y=y, groups=groups, repeats=repeats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5,
                    help="CV repeats per variant (harness default 5)")
    ap.add_argument("--jobs", type=int, default=8, help="variants evaluated in parallel")
    args = ap.parse_args()

    df, cols, X, y, groups = load_dataset()
    print(f"ข้อมูล: {len(df)} หัว | {len(cols)} ฟีเจอร์ | "
          f"{int((y == 1).sum())} พบเชื้อ / {int((y == 0).sum())} ไม่พบ")
    print(f"repeats={args.repeats} x 5-fold = {args.repeats * 5} โฟลด์ต่อวิธี\n", flush=True)

    variants = build_variants()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT_CSV.with_name(OUT_CSV.stem + "_partial.csv")
    partial.unlink(missing_ok=True)

    results = []
    stream = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_run_one)(i, X, y, groups, args.repeats) for i in range(len(variants))
    )
    for r in stream:
        results.append(r)
        s = r["summary"]
        # append as we go: a killed run still leaves everything finished so far
        s.to_csv(partial, mode="a", header=not partial.exists(), index=False)
        print(f"[{len(results)}/{len(variants)}] {r['name']:<42} "
              f"acc={s['accuracy'].iloc[0]:.3f} rec={s['recall'].iloc[0]:.3f} "
              f"spec={s['specificity'].iloc[0]:.3f} kappa={s['kappa'].iloc[0]:.3f} "
              f"auc={s['auc'].iloc[0]:.3f}", flush=True)

    table = compare(results, sort_by="kappa")
    deltas = paired_deltas(results, "baseline RF thr0.5")
    table = table.merge(deltas, on="name", how="left")

    print("\n=== ตารางเปรียบเทียบ (เรียงตาม kappa) ===")
    print_table(table)

    print("\n=== ส่วนต่างแบบจับคู่รายโฟลด์เทียบ baseline (kappa) ===")
    show = table[["name", "kappa", "d_kappa_mean", "d_kappa_sd", "d_kappa_sem",
                  "win_folds", "lose_folds"]].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n=== SD ระหว่างโฟลด์ (ตัวเทียบว่าส่วนต่างอยู่ในระดับ noise หรือไม่) ===")
    sd_cols = ["name"] + [f"{m}_sd" for m in METRICS]
    print(table[sd_cols].round(3).to_string(index=False))

    table["repeats"] = args.repeats
    table.to_csv(OUT_CSV, index=False)
    partial.unlink(missing_ok=True)
    print(f"\nบันทึกตารางเต็มที่ {OUT_CSV}")


if __name__ == "__main__":
    main()
