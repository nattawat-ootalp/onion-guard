"""How many features should the model keep, and is bad data holding the score down?

WHAT THIS ANSWERS
  1. Feature-set size. k is swept 3..24 with the ranking redone INSIDE every
     training fold (honest) and, for the same k, with the ranking done once on
     all 60 heads and then frozen (leaky, matching how
     data/selected_features.json was built). The difference between the two
     curves IS the selection-leakage gap, measured instead of assumed.
  2. Two other in-fold selectors: recursive feature elimination and L1
     (logistic-lasso) selection, so "top-k by RF importance" is not the only
     thing on trial.
  3. Data quality. S045's onion detection failed and its features come from a
     fallback centre crop; the run reports what dropping it does to the mean
     AND to the fold-to-fold SD. A statistical audit (duplicates, robust
     z-scores, MinCovDet distance, constant columns) runs first and prints
     what it finds, so any further row drop is argued from the data rather
     than from the score.
  4. uv_exclusive_dot_frac, the cross-check against a visible-light photo of
     the same head, which the pinned 10 leave out.

HOW TO READ THE OUTPUT
Every number comes from src/eval_harness.py: StratifiedGroupKFold(5) x 5
repeats, threshold fixed at 0.5, mean over all 25 folds. With 12 heads per
test fold the SD of kappa across folds is ~0.29, so a 0.03 difference in mean
kappa is noise. Because every variant that keeps all 60 rows sees exactly the
same 25 splits, the table also reports the PAIRED per-fold difference against
the pinned-10 baseline (delta_kappa +/- SE), which is the only comparison here
sensitive enough to separate a real effect from fold-assignment luck.

Nothing in this script writes to data/, models/ or config.json. It only reads
features.csv and writes reports/experiments/exp_featureset.csv.

    PYTHONPATH=src .venv/bin/python src/experiments/exp_featureset.py
    PYTHONPATH=src .venv/bin/python src/experiments/exp_featureset.py --repeats 3 --jobs 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import MinCovDet
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import ROOT
from eval_harness import (
    FEATURES_PATH, ID_COL, LABEL_COL, SELECTED_PATH, compare, evaluate_method,
    print_table,
)

OUT_DIR = ROOT / "reports" / "experiments"
OUT_CSV = OUT_DIR / "exp_featureset.csv"

UV_FEATURE = "uv_exclusive_dot_frac"

# Heads whose framing is known to be unreliable. NOT chosen by looking at the
# model's errors: this is the pipeline's own record. Supabase table `scans`
# carries framing_ok and quality_notes.framing_failed_views for every scan;
# queried 2026-07-29 across the 60 labelled heads, exactly one row is flagged
# (S045, framing_failed_views == ["V1"]), and every head is ood=in_range with
# consistent EXIF. So "drop the unreliable framing" and "drop S045" are the
# same experiment on today's data.
FRAMING_SUSPECT = ["S045"]


# --------------------------------------------------------------------------
# methods. Each is a module-level callable (picklable, so joblib can fan the
# variants out across cores) matching the harness contract:
#     fit_fn(X_train, y_train, seed) -> (scorer, threshold)
# Anything that looks at y happens on the training split only.
# --------------------------------------------------------------------------

def _rf(seed, n_estimators=400):
    # n_jobs=1: on 48x24 the thread pool costs more than it saves, and the
    # fitted forest is identical either way for a fixed random_state.
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1)


class _SubsetScorer:
    """Picklable replacement for `lambda X: model.predict_proba(X[:, idx])`."""

    def __init__(self, model, idx):
        self.model, self.idx = model, idx

    def __call__(self, X):
        return self.model.predict_proba(X[:, self.idx])[:, 1]


class FixedColumns:
    """Train on a column subset decided before CV started.

    Honest when the subset was fixed for a reason unrelated to the labels
    (e.g. "all columns", "all but uv"); LEAKY when the subset was ranked on
    all 60 heads, which is exactly what this experiment wants to measure.
    """

    def __init__(self, idx):
        self.idx = np.asarray(idx, dtype=int)

    def __call__(self, X_train, y_train, seed):
        model = _rf(seed).fit(X_train[:, self.idx], y_train)
        return _SubsetScorer(model, self.idx), 0.5


class InFoldTopK:
    """Rank by RF importance on the training fold, keep the top k, refit."""

    def __init__(self, k, record=None):
        self.k, self.record = k, record

    def __call__(self, X_train, y_train, seed):
        ranker = _rf(seed).fit(X_train, y_train)
        idx = np.argsort(ranker.feature_importances_)[::-1][:self.k]
        if self.record is not None:
            self.record.append(idx)
        model = _rf(seed).fit(X_train[:, idx], y_train)
        return _SubsetScorer(model, idx), 0.5


class InFoldRFE:
    """Recursive feature elimination inside the training fold."""

    def __init__(self, k, step=1, record=None):
        self.k, self.step, self.record = k, step, record

    def __call__(self, X_train, y_train, seed):
        sel = RFE(_rf(seed, n_estimators=200), n_features_to_select=self.k,
                  step=self.step).fit(X_train, y_train)
        idx = np.flatnonzero(sel.support_)
        if self.record is not None:
            self.record.append(idx)
        model = _rf(seed).fit(X_train[:, idx], y_train)
        return _SubsetScorer(model, idx), 0.5


class InFoldL1:
    """L1 logistic regression picks the columns; RF does the classifying.

    Scaling is fitted on the training fold only — a StandardScaler fitted on
    all 60 rows would leak the test fold's spread into the selection.
    """

    def __init__(self, C, record=None):
        self.C, self.record = C, record

    def __call__(self, X_train, y_train, seed):
        scaler = StandardScaler().fit(X_train)
        # l1_ratio=1.0 is sklearn >=1.8's spelling of penalty="l1" (checked:
        # both select the same 8 columns at C=0.5 on this data).
        lr = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=self.C,
                                max_iter=5000, random_state=seed)
        lr.fit(scaler.transform(X_train), y_train)
        idx = np.flatnonzero(np.abs(lr.coef_[0]) > 1e-10)
        if idx.size == 0:  # lasso zeroed everything: fall back to all columns
            idx = np.arange(X_train.shape[1])
        if self.record is not None:
            self.record.append(idx)
        model = _rf(seed).fit(X_train[:, idx], y_train)
        return _SubsetScorer(model, idx), 0.5


def global_rank_idx(X, y, k, seed):
    """Top-k ranked on the WHOLE dataset — deliberately leaky, used only as
    the comparison arm that quantifies how optimistic that procedure is."""
    ranker = _rf(seed).fit(X, y)
    return np.argsort(ranker.feature_importances_)[::-1][:k]


# --------------------------------------------------------------------------
# data-quality audit
# --------------------------------------------------------------------------

def audit_rows(df, feature_cols):
    """Report suspect rows BEFORE any of them is dropped.

    Deliberately blind to the model: nothing here looks at whether a row is
    predicted correctly. Dropping rows the model gets wrong would fit the
    evaluation, not the data.
    """
    X = df[feature_cols].values.astype(float)
    print("\n=== ตรวจคุณภาพข้อมูล (data-quality audit) ===")
    print(f"rows={len(df)}  features={len(feature_cols)}  "
          f"pos={int((df[LABEL_COL] == 1).sum())}  neg={int((df[LABEL_COL] == 0).sum())}")

    dup_id = df[ID_COL].duplicated().sum()
    dup_feat = df.duplicated(subset=feature_cols).sum()
    print(f"duplicate sample_code: {dup_id} | duplicate feature rows: {dup_feat}")
    print(f"missing values: {int(df[feature_cols].isna().sum().sum())}")

    const = [c for c in feature_cols if df[c].nunique() <= 1]
    print(f"constant columns (carry no information): {const or 'none'}")

    # Robust z: median/MAD, so one wild row cannot hide itself by inflating
    # the SD it is measured against.
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs((X - med) / np.where(mad > 0, mad, np.nan))
    zmax = np.nanmax(z, axis=1)
    worst = [feature_cols[i] for i in np.nanargmax(np.nan_to_num(z), axis=1)]

    scaled = StandardScaler().fit_transform(X)
    try:
        dist = MinCovDet(random_state=0, support_fraction=0.9).fit(scaled).mahalanobis(scaled)
    except Exception:  # pragma: no cover - singular covariance on small n
        dist = np.full(len(df), np.nan)

    audit = pd.DataFrame({
        "sample_code": df[ID_COL].values,
        "label": df[LABEL_COL].values,
        "max_robust_z": zmax,
        "worst_feature": worst,
        "n_feat_z_gt5": np.nansum(z > 5, axis=1),
        "mcd_dist": dist,
        "framing_flagged": df[ID_COL].isin(FRAMING_SUSPECT).values,
    })
    print("\nrows ranked by robust z (top 8):")
    print(audit.sort_values("max_robust_z", ascending=False).head(8).to_string(index=False))
    print("\nrows ranked by MinCovDet distance (top 6):")
    print(audit.sort_values("mcd_dist", ascending=False).head(6).to_string(index=False))
    print("\nrows flagged by the pipeline itself (scans.framing_ok is false):")
    print(audit[audit["framing_flagged"]].to_string(index=False))
    return audit


# --------------------------------------------------------------------------
# variant plumbing
# --------------------------------------------------------------------------

def run_variant(spec, data, repeats):
    """spec: dict with name/fit/rows/meta. Returns the harness result + meta."""
    X, y, groups = data[spec["rows"]]
    res = evaluate_method(spec["fit"], name=spec["name"], X=X, y=y, groups=groups,
                          repeats=repeats)
    res["meta"] = spec["meta"]
    res["rows_key"] = spec["rows"]
    return res


def paired_delta(res, ref, metric):
    """Mean per-fold difference vs the reference, and its standard error.

    Only meaningful when both ran on the same rows: the harness derives its
    splits from y and groups, so identical rows + identical seeds means fold i
    of repeat r contains exactly the same heads in both runs.
    """
    a = res["folds"].sort_values(["repeat", "fold"])[metric].values
    b = ref["folds"].sort_values(["repeat", "fold"])[metric].values
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def introspect(rows, all_cols, pinned, repeats):
    """Which columns the in-fold selectors actually keep, fold by fold.

    Re-runs the two selectors in-process (same seeds -> same folds -> same
    picks as the scored run) because joblib workers cannot hand mutated state
    back. A feature picked in 25/25 folds is a stable signal; one picked in
    9/25 is the fold assignment talking.
    """
    X, y, groups = rows
    for label, method, rec in (
        ("in-fold top-10 (RF importance)", InFoldTopK, []),
        ("L1 in-fold (C=0.5)", InFoldL1, []),
    ):
        fit = InFoldTopK(10, record=rec) if method is InFoldTopK else InFoldL1(0.5, record=rec)
        evaluate_method(fit, name="record", X=X, y=y, groups=groups, repeats=repeats)
        counts = pd.Series([all_cols[i] for idx in rec for i in idx]).value_counts()
        sizes = [len(idx) for idx in rec]
        print(f"\n=== ฟีเจอร์ที่ {label} เลือก (จาก {len(rec)} folds) ===")
        print(f"ขนาดชุดฟีเจอร์ต่อ fold: mean {np.mean(sizes):.1f}  min {min(sizes)}  max {max(sizes)}")
        for name, n in counts.items():
            mark = " (in pinned10)" if name in pinned else ""
            print(f"  {name:26s} {n:>3}/{len(rec)} folds{mark}")
        print(f"pinned10 features this method never picks: "
              f"{[c for c in pinned if c not in counts.index] or 'none'}")
        print(f"picked in EVERY fold: {[c for c, n in counts.items() if n == len(rec)]}")


def main():
    ap = argparse.ArgumentParser(description="หาจำนวนฟีเจอร์ที่ดีที่สุด + ตรวจปัญหาคุณภาพข้อมูล")
    ap.add_argument("--introspect-only", action="store_true",
                    help="ข้ามการวัดผลทั้งหมด พิมพ์เฉพาะฟีเจอร์ที่ in-fold selection เลือก")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=24)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    all_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    pinned = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))["features"]
    col_idx = {c: i for i, c in enumerate(all_cols)}

    audit = audit_rows(df, all_cols)
    audit.to_csv(OUT_DIR / "exp_featureset_row_audit.csv", index=False)

    # The two rows the audit puts furthest from the rest of the distribution.
    # Reported as a sensitivity check only — being an outlier is not by itself
    # evidence of a measurement error, and neither row is flagged by the
    # capture pipeline.
    stat_outliers = list(audit.sort_values("mcd_dist", ascending=False)["sample_code"].head(2))

    def subset(drop):
        m = ~df[ID_COL].isin(drop).values
        return (df.loc[m, all_cols].values, df.loc[m, LABEL_COL].values,
                df.loc[m, ID_COL].values)

    data = {
        "full": subset([]),
        "no_s045": subset(FRAMING_SUSPECT),
        "no_s045_outliers": subset(FRAMING_SUSPECT + stat_outliers),
    }
    print(f"\nrow subsets: full={len(data['full'][1])}  "
          f"no_s045={len(data['no_s045'][1])} (drop {FRAMING_SUSPECT})  "
          f"no_s045_outliers={len(data['no_s045_outliers'][1])} "
          f"(also drop {stat_outliers}, diagnostic only)")

    if args.introspect_only:
        introspect(data["full"], all_cols, pinned, args.repeats)
        return 0

    pinned_idx = [col_idx[c] for c in pinned]
    pinned_uv_idx = pinned_idx + [col_idx[UV_FEATURE]]
    no_uv_idx = [i for c, i in col_idx.items() if c != UV_FEATURE]
    ks = list(range(args.k_min, min(args.k_max, len(all_cols)) + 1))

    specs = []

    def add(name, fit, rows="full", **meta):
        specs.append({"name": name, "fit": fit, "rows": rows, "meta": meta})

    # --- reference points -------------------------------------------------
    add("pinned10 fixed (BASELINE, leaky selection)", FixedColumns(pinned_idx),
        family="reference", k=10, selection="fixed-pinned")
    add("all 24 features", FixedColumns(list(range(len(all_cols)))),
        family="reference", k=len(all_cols), selection="none")

    # --- 1. k sweep, in-fold ranking vs whole-data ranking ----------------
    seed_for_global = 42  # any fixed seed; the leak is the point, not the seed
    for k in ks:
        add(f"in-fold top-{k:02d} (RF importance)", InFoldTopK(k),
            family="k-sweep-infold", k=k, selection="in-fold RF importance")
        gidx = global_rank_idx(df[all_cols].values, df[LABEL_COL].values, k, seed_for_global)
        add(f"whole-data top-{k:02d} (LEAKY)", FixedColumns(gidx),
            family="k-sweep-leaky", k=k, selection="ranked on all 60 heads")

    # --- 2. RFE, 3. L1 ----------------------------------------------------
    for k in (5, 8, 10, 12, 15):
        add(f"RFE in-fold -> {k}", InFoldRFE(k), family="rfe", k=k,
            selection="in-fold RFE")
    for C in (0.05, 0.1, 0.5, 1.0):
        add(f"L1 in-fold (C={C})", InFoldL1(C), family="l1", k=np.nan,
            selection="in-fold L1 logistic")

    # --- 4. row drops -----------------------------------------------------
    for rows, tag in (("no_s045", "drop S045 (bad framing)"),
                      ("no_s045_outliers", f"drop S045+{'+'.join(stat_outliers)}")):
        add(f"pinned10 fixed, {tag}", FixedColumns(pinned_idx), rows=rows,
            family="row-drop", k=10, selection="fixed-pinned")
        add(f"in-fold top-10, {tag}", InFoldTopK(10), rows=rows,
            family="row-drop", k=10, selection="in-fold RF importance")
        add(f"all 24 features, {tag}", FixedColumns(list(range(len(all_cols)))),
            rows=rows, family="row-drop", k=len(all_cols), selection="none")

    # --- 6. uv_exclusive_dot_frac ----------------------------------------
    add(f"pinned10 + {UV_FEATURE}", FixedColumns(pinned_uv_idx),
        family="uv", k=11, selection="fixed-pinned + uv")
    add(f"all 23 features (no {UV_FEATURE})", FixedColumns(no_uv_idx),
        family="uv", k=23, selection="none, uv removed")
    for k in (5, 8, 10, 12):
        # Same in-fold procedure, but uv is not even in the pool. Compared
        # against "in-fold top-k" above, this isolates what uv contributes.
        add(f"in-fold top-{k:02d}, no-uv pool", _NoUvPool(k, no_uv_idx),
            family="uv", k=k, selection="in-fold RF importance, uv excluded")

    print(f"\nrunning {len(specs)} variants x {args.repeats} repeats x 5 folds ...")
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_variant)(s, data, args.repeats) for s in specs)

    by_name = {r["name"]: r for r in results}
    ref = by_name["pinned10 fixed (BASELINE, leaky selection)"]

    table = compare(results)
    meta = pd.DataFrame([{"name": r["name"], "rows_key": r["rows_key"],
                          "n_rows": len(data[r["rows_key"]][1]), **r["meta"]}
                         for r in results])
    table = table.merge(meta, on="name", how="left")

    # Paired deltas: only defined against a reference that saw the same rows.
    deltas = []
    for r in results:
        if r["rows_key"] == "full":
            dk, sek = paired_delta(r, ref, "kappa")
            da, sea = paired_delta(r, ref, "accuracy")
        else:
            dk = sek = da = sea = np.nan
        deltas.append({"name": r["name"], "delta_kappa_vs_baseline": dk,
                       "delta_kappa_se": sek, "delta_accuracy_vs_baseline": da,
                       "delta_accuracy_se": sea})
    table = table.merge(pd.DataFrame(deltas), on="name", how="left")
    table.to_csv(OUT_CSV, index=False)

    print("\n=== ตารางเปรียบเทียบทั้งหมด (เรียงตาม kappa) ===")
    print_table(table)

    print("\n=== ช่องว่างจากการเลือกฟีเจอร์แบบรั่ว (selection-leakage gap) ===")
    print("เลือกฟีเจอร์จากข้อมูลทั้ง 60 หัวก่อนแล้วค่อย CV = มองโลกในแง่ดีเกินจริงเท่าไร")
    gap_rows = []
    for k in ks:
        a = by_name[f"in-fold top-{k:02d} (RF importance)"]["summary"].iloc[0]
        b = by_name[f"whole-data top-{k:02d} (LEAKY)"]["summary"].iloc[0]
        gap_rows.append({"k": k, "kappa_infold": a["kappa"], "kappa_leaky": b["kappa"],
                         "kappa_gap": b["kappa"] - a["kappa"],
                         "acc_infold": a["accuracy"], "acc_leaky": b["accuracy"],
                         "acc_gap": b["accuracy"] - a["accuracy"]})
    gap = pd.DataFrame(gap_rows)
    print(gap.round(3).to_string(index=False))
    print(f"\nmean kappa gap over k={ks[0]}..{ks[-1]}: {gap['kappa_gap'].mean():+.3f} | "
          f"mean accuracy gap: {gap['acc_gap'].mean():+.3f}")
    pin = ref["summary"].iloc[0]
    infold10 = by_name.get("in-fold top-10 (RF importance)")
    if infold10 is not None:
        i10 = infold10["summary"].iloc[0]
        dk, sek = paired_delta(infold10, ref, "kappa")
        print(f"pinned10 (the number the report currently quotes) kappa {pin['kappa']:.3f} "
              f"vs in-fold top-10 kappa {i10['kappa']:.3f} "
              f"-> gap {pin['kappa'] - i10['kappa']:+.3f} "
              f"(paired per-fold {-dk:+.3f} +/- {sek:.3f} SE)")
    gap.to_csv(OUT_DIR / "exp_featureset_leakage_gap.csv", index=False)

    print("\n=== ความแปรปรวนและผลต่างแบบจับคู่รายโฟลด์ (ตัวเลขที่ใช้ตัดสินจริง) ===")
    print("delta = ผลต่างจาก baseline บนโฟลด์เดียวกัน (+/- SE). |delta| < 2*SE = แยกจาก noise ไม่ได้")
    head = table.sort_values("kappa", ascending=False).head(15)
    cols = ["name", "n_rows", "kappa", "kappa_sd", "delta_kappa_vs_baseline",
            "delta_kappa_se", "accuracy", "accuracy_sd", "delta_accuracy_vs_baseline"]
    print(head[cols].round(3).to_string(index=False))

    print("\n=== ผลของการตัดแถวที่มีปัญหา (mean และ SD) ===")
    rowdrop = table[table["family"].isin(["row-drop"]) |
                    table["name"].isin([ref["name"], "all 24 features",
                                        "in-fold top-10 (RF importance)"])]
    print(rowdrop[["name", "n_rows", "accuracy", "accuracy_sd", "kappa", "kappa_sd",
                   "recall", "specificity", "auc"]].round(3).to_string(index=False))

    introspect(data["full"], all_cols, pinned, args.repeats)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_DIR / 'exp_featureset_leakage_gap.csv'}")
    print(f"Wrote {OUT_DIR / 'exp_featureset_row_audit.csv'}")


class _NoUvPool:
    """in-fold top-k restricted to a fixed column pool (used to drop uv).

    The pool restriction is label-free, so it is not leakage; the ranking
    inside it still happens on the training fold only.
    """

    def __init__(self, k, pool_idx):
        self.k, self.pool = k, np.asarray(pool_idx, dtype=int)

    def __call__(self, X_train, y_train, seed):
        Xp = X_train[:, self.pool]
        ranker = _rf(seed).fit(Xp, y_train)
        top = np.argsort(ranker.feature_importances_)[::-1][:self.k]
        idx = self.pool[top]
        model = _rf(seed).fit(X_train[:, idx], y_train)
        return _SubsetScorer(model, idx), 0.5


if __name__ == "__main__":
    sys.exit(main())
