"""Experiment: does making the features robust to shot-to-shot exposure
variation improve classification?

WHY
The 60 heads were shot hand-held with a phone, no jig. Several of the
strongest features (sd_G, mean_G, sd_B, texture) are ABSOLUTE brightness
statistics in linear light, so a change of exposure/distance multiplies them
all by roughly the same factor. A measured probe put a +/-15% brightness
change at ~0.35 dataset SD in feature space. This script tests whether
removing that nuisance direction helps, hurts, or does nothing.

TWO CATEGORIES OF TRANSFORM (stated explicitly, per the protocol)
  A) ROW-WISE ARITHMETIC — computed from the values of ONE row only
     (dividing a column by another column of the same row, log1p, or
     standardising a row against its own colour mean/spread). No parameter
     is estimated from the dataset, so there is nothing to leak and these are
     applied up front, before the CV split. Functions below are named
     rowwise_*.
  B) DATA-FITTED — QuantileTransformer, StandardScaler, PCA. These estimate
     parameters from data and are therefore ALWAYS inside an sklearn Pipeline
     so eval_harness fits them on the training fold only. Functions below are
     named method_* and build the Pipeline inside fit_fn.

Everything is scored through src/eval_harness.py unchanged.

NOTE ON RANDOM FORESTS AND MONOTONE TRANSFORMS
A decision tree splits on "feature <= t". Any strictly increasing per-feature
transform (log1p, per-feature quantile/rank) maps t to f(t) and leaves the
partition of the rows identical, so an RF is mathematically invariant to it.
Those variants are still run (they are on the mandated list) but they are
expected to reproduce the baseline exactly; where they matter is for a linear
model, so logistic-regression counterparts are run alongside. Transforms that
MIX columns (ratios, per-row standardisation, PCA) are not monotone per
feature and genuinely change what the forest can see.
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from common import ROOT
from eval_harness import compare, evaluate_method, print_table

FEATURES_PATH = ROOT / "data" / "features.csv"
SELECTED_PATH = ROOT / "data" / "selected_features.json"
OUT_DIR = ROOT / "reports" / "experiments"
OUT_PATH = OUT_DIR / "exp_features.csv"

ID_COL, LABEL_COL = "sample_code", "compactdry"

# ---------------------------------------------------------------------------
# Feature taxonomy, read off src/extract_features.py
# ---------------------------------------------------------------------------
# Absolute intensity: computed from linear-light channel values, so an
# exposure change of factor f multiplies each of these by ~f. texture is the
# SD of a Laplacian of brightness, which is linear in brightness, so it
# scales too. These are the nuisance-carrying block.
ABS_INTENSITY = [
    "mean_R", "mean_G", "mean_B",
    "sd_R", "sd_G", "sd_B",
    "brightness_mean", "brightness_sd",
    "texture", "F_p95", "F_p05",
]

# The six-column colour block used by the per-row standardisation variant.
COLOUR_BLOCK = ["mean_R", "mean_G", "mean_B", "sd_R", "sd_G", "sd_B"]

# Already scale-free by construction:
#   A_high/A_low  - fraction of ROI beyond (own median +/- k * own MAD);
#                   median and MAD both scale with f, so the fraction does not
#   NDFI_*        - (G - avg(R,B)) / (G + avg(R,B)), a ratio
#   blob_*/n_*/diam/density/ratio - pixel counts, geometry and count ratios
#   uv_exclusive_dot_frac - a fraction
SCALE_FREE = [
    "A_high", "A_low",
    "NDFI_mean", "NDFI_p95", "NDFI_p05",
    "blob_max",
    "n_small_sharp_blobs", "n_large_blotches",
    "avg_small_blob_diam_px", "avg_large_blotch_diam_px",
    "cluster_density", "ratio_small_to_large",
    "uv_exclusive_dot_frac",
]

# Heavy-tailed counts / areas / geometry -> candidates for log1p.
HEAVY_TAILED = [
    "blob_max", "n_small_sharp_blobs", "n_large_blotches",
    "avg_small_blob_diam_px", "avg_large_blotch_diam_px",
    "A_high", "A_low", "cluster_density",
]


def load_df():
    return pd.read_csv(FEATURES_PATH)


def selected_cols():
    return json.loads(SELECTED_PATH.read_text(encoding="utf-8"))["features"]


# ---------------------------------------------------------------------------
# CATEGORY A: row-wise arithmetic (no fitted parameters -> safe up front)
# ---------------------------------------------------------------------------
def rowwise_ratio(df, cols, ref="brightness_mean", keep_ref=False):
    """Divide every absolute-intensity column in `cols` by that same row's
    intensity reference. Leaves scale-free columns untouched.

    Pure per-row arithmetic: value_ij / value_i_ref. Nothing is estimated
    across rows.
    """
    out = pd.DataFrame(index=df.index)
    names = []
    for c in cols:
        if c in ABS_INTENSITY and c != ref:
            out[f"{c}/{ref}"] = df[c].values / np.maximum(df[ref].values, 1e-9)
            names.append(f"{c}/{ref}")
        elif c == ref:
            if keep_ref:
                out[c] = df[c].values
                names.append(c)
        else:
            out[c] = df[c].values
            names.append(c)
    if keep_ref and ref not in names:
        # Keep one column carrying the overall exposure level, so the model
        # can still use absolute brightness if it is genuinely informative.
        out[ref] = df[ref].values
        names.append(ref)
    return out[names]


def rowwise_log(df, cols):
    """log1p on the heavy-tailed count/area columns. Per-row, per-cell."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in HEAVY_TAILED:
            out[f"log1p({c})"] = np.log1p(df[c].values)
        else:
            out[c] = df[c].values
    return out


def rowwise_colour_zscore(df, cols):
    """Replace the colour block by its per-ROW z-scores.

    For each row: m = mean of the 6 colour features of that row, s = their SD;
    each colour feature becomes (x - m) / s. This kills both the row's overall
    brightness level AND its overall spread, leaving only the SHAPE of the
    colour profile. m and s come from the row itself, never from the dataset.
    Any colour column requested in `cols` is replaced by its z-scored version;
    other columns pass through.
    """
    block = df[COLOUR_BLOCK].values
    m = block.mean(axis=1, keepdims=True)
    s = block.std(axis=1, keepdims=True)
    z = (block - m) / np.maximum(s, 1e-12)
    zdf = pd.DataFrame(z, columns=[f"z_{c}" for c in COLOUR_BLOCK], index=df.index)

    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in COLOUR_BLOCK:
            out[f"z_{c}"] = zdf[f"z_{c}"].values
        else:
            out[c] = df[c].values
    return out


def perturb_exposure(df, rng, pct=0.15):
    """Simulate the nuisance we are trying to defend against.

    Gives every head its own exposure factor f (uniform in [1-pct, 1+pct])
    and multiplies the absolute-intensity block by it, leaving the scale-free
    block alone. This is the first-order model of a hand-held exposure /
    distance change in linear light. It is a STRESS TEST applied to the whole
    matrix before evaluation, not a training-time augmentation, and it is
    reported separately from the main table.
    """
    out = df.copy()
    f = rng.uniform(1 - pct, 1 + pct, size=len(df))
    for c in ABS_INTENSITY:
        out[c] = df[c].values * f
    return out


# ---------------------------------------------------------------------------
# CATEGORY B: methods (anything data-fitted lives in a Pipeline)
# ---------------------------------------------------------------------------
def method_rf(X_train, y_train, seed):
    # n_jobs=1: with 48 training rows the joblib thread pool costs far more
    # than it saves. Predictions are identical to the harness's n_jobs=-1
    # baseline (verified: same accuracy 0.793 / kappa 0.560).
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    return (lambda X: rf.predict_proba(X)[:, 1]), 0.5


def method_lr(X_train, y_train, seed):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
    ]).fit(X_train, y_train)
    return (lambda X: pipe.predict_proba(X)[:, 1]), 0.5


def _n_quantiles(n_train):
    return max(10, min(100, n_train))


def method_quantile_rf(X_train, y_train, seed):
    """QuantileTransformer FITTED INSIDE THE FOLD (Pipeline), then RF."""
    pipe = Pipeline([
        ("qt", QuantileTransformer(n_quantiles=_n_quantiles(len(X_train)),
                                   output_distribution="normal",
                                   random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)),
    ]).fit(X_train, y_train)
    return (lambda X: pipe.predict_proba(X)[:, 1]), 0.5


def method_quantile_lr(X_train, y_train, seed):
    pipe = Pipeline([
        ("qt", QuantileTransformer(n_quantiles=_n_quantiles(len(X_train)),
                                   output_distribution="normal",
                                   random_state=seed)),
        ("lr", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
    ]).fit(X_train, y_train)
    return (lambda X: pipe.predict_proba(X)[:, 1]), 0.5


def make_pca_rf(n_components):
    """StandardScaler + PCA FITTED INSIDE THE FOLD, then RF."""
    def fit_fn(X_train, y_train, seed):
        k = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=k, random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)),
        ]).fit(X_train, y_train)
        return (lambda X: pipe.predict_proba(X)[:, 1]), 0.5
    return fit_fn


def make_pca_drop1_rf(n_components):
    """Scaler + PCA in-fold, then DROP PC1 before the forest.

    On a matrix dominated by co-scaling brightness features, PC1 is close to
    the "overall exposure" direction. Dropping it is the fitted-transform
    analogue of the row-wise ratio. Fitted in-fold, so no leakage.
    """
    def fit_fn(X_train, y_train, seed):
        k = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
        sc = StandardScaler().fit(X_train)
        pca = PCA(n_components=k, random_state=seed).fit(sc.transform(X_train))
        rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)
        rf.fit(pca.transform(sc.transform(X_train))[:, 1:], y_train)

        def scorer(X):
            return rf.predict_proba(pca.transform(sc.transform(X))[:, 1:])[:, 1]
        return scorer, 0.5
    return fit_fn


# ---------------------------------------------------------------------------
# Variant table
# ---------------------------------------------------------------------------
def build_variants(df):
    """(name, X, method) for every variant. X is already row-wise transformed;
    anything data-fitted is inside the method's Pipeline."""
    sel = selected_cols()
    allc = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]

    intensity_free = [c for c in allc if c not in ABS_INTENSITY]
    intensity_only = [c for c in allc if c in ABS_INTENSITY]
    sel_free = [c for c in sel if c not in ABS_INTENSITY]

    V = []
    add = lambda name, X, m: V.append((name, np.asarray(X, dtype=float), m))

    # --- references -------------------------------------------------------
    add("0  baseline RF sel10", df[sel], method_rf)
    add("0b baseline RF all24", df[allc], method_rf)
    add("0c baseline LR sel10", df[sel], method_lr)
    # Control for variant 1d: 1d both divides by brightness_mean AND hands the
    # model brightness_mean itself, which is a column the selected-10 set never
    # had. Without this control a gain from "one extra feature" would be
    # misread as a gain from "the ratio transform".
    add("0d sel10 + brightness_mean", df[sel + ["brightness_mean"]], method_rf)

    # --- 1. row-wise ratio features --------------------------------------
    add("1a ratio/brightness sel10", rowwise_ratio(df, sel, "brightness_mean"), method_rf)
    add("1b ratio/brightness all24", rowwise_ratio(df, allc, "brightness_mean"), method_rf)
    add("1c ratio/mean_B sel10", rowwise_ratio(df, sel, "mean_B"), method_rf)
    add("1d ratio/brightness +ref sel10",
        rowwise_ratio(df, sel, "brightness_mean", keep_ref=True), method_rf)
    add("1e ratio/brightness sel10 LR", rowwise_ratio(df, sel, "brightness_mean"), method_lr)

    # --- 2. log transform of heavy-tailed counts/areas --------------------
    add("2a log counts sel10", rowwise_log(df, sel), method_rf)
    add("2b log counts all24", rowwise_log(df, allc), method_rf)
    add("2c log counts sel10 LR", rowwise_log(df, sel), method_lr)

    # --- 3. per-row standardisation of the colour block -------------------
    add("3a rowz colour sel10", rowwise_colour_zscore(df, sel), method_rf)
    add("3b rowz colour all24", rowwise_colour_zscore(df, allc), method_rf)

    # --- 4. rank/quantile transform fitted in-fold ------------------------
    add("4a quantile(in-fold) RF sel10", df[sel], method_quantile_rf)
    add("4b quantile(in-fold) LR sel10", df[sel], method_quantile_lr)
    add("4c quantile(in-fold) LR all24", df[allc], method_quantile_lr)

    # --- 5. drop absolute intensity entirely ------------------------------
    add("5a intensity-free all", df[intensity_free], method_rf)
    add("5b intensity-free sel10", df[sel_free], method_rf)
    add("5c intensity-ONLY (diagnostic)", df[intensity_only], method_rf)

    # --- 6. PCA fitted in-fold, plain and after the ratio transform -------
    add("6a PCA(in-fold) all24", df[allc], make_pca_rf(10))
    add("6b PCA(in-fold) drop PC1 all24", df[allc], make_pca_drop1_rf(10))
    add("6c PCA(in-fold) on ratio all24",
        rowwise_ratio(df, allc, "brightness_mean"), make_pca_rf(10))
    add("6d PCA(in-fold) sel10", df[sel], make_pca_rf(8))

    # --- 7. combinations --------------------------------------------------
    ratio_all = rowwise_ratio(df, allc, "brightness_mean")
    add("7a ratio+log all24", rowwise_log(ratio_all, list(ratio_all.columns)), method_rf)
    ratio_sel = rowwise_ratio(df, sel, "brightness_mean")
    add("7b ratio+log sel10", rowwise_log(ratio_sel, list(ratio_sel.columns)), method_rf)
    add("7c ratio+log+quantile sel10 LR",
        rowwise_log(ratio_sel, list(ratio_sel.columns)), method_quantile_lr)
    free_log = rowwise_log(df[intensity_free], intensity_free)
    add("7d intensity-free + log", free_log, method_rf)
    # Scale-free block plus ONE relative-contrast column (contrast is a ratio,
    # so it survives an exposure change) - the "keep the physics, drop the
    # nuisance" combination.
    contrast = pd.DataFrame({
        "sd_B/mean_B": df["sd_B"].values / np.maximum(df["mean_B"].values, 1e-9),
        "sd_G/mean_G": df["sd_G"].values / np.maximum(df["mean_G"].values, 1e-9),
        "mean_G/mean_B": df["mean_G"].values / np.maximum(df["mean_B"].values, 1e-9),
        "texture/brightness": df["texture"].values / np.maximum(df["brightness_mean"].values, 1e-9),
    }, index=df.index)
    add("7e intensity-free + contrast",
        pd.concat([df[intensity_free], contrast], axis=1), method_rf)
    add("7f sel10 + contrast", pd.concat([df[sel], contrast], axis=1), method_rf)

    return V


def run(df, y, groups, tag="", quiet=False):
    results = []
    for name, X, method in build_variants(df):
        results.append(evaluate_method(method, name=name, X=X, y=y, groups=groups))
        if not quiet:
            s = results[-1]["summary"].iloc[0]
            print(f"  {name:34s} acc={s.accuracy:.3f} kappa={s.kappa:.3f} auc={s.auc:.3f}")
    table = compare(results)
    if tag:
        table.insert(1, "condition", tag)
    return table, results


BASELINE_NAME = "0  baseline RF sel10"


def paired_vs_baseline(results, metric="kappa"):
    """Fold-by-fold paired comparison against the baseline.

    StratifiedGroupKFold.split() depends only on y, groups and random_state —
    never on the VALUES in X — and evaluate_method uses the same seeds for
    every variant. So fold i of repeat r contains exactly the same heads in
    every variant, and the differences can be paired. Pairing removes the
    fold-assignment variance, which is the dominant term here (kappa_sd ~0.2
    between folds), and is the only way to see whether a +0.03 mean difference
    is a real effect or the same folds being re-shuffled.
    """
    base = {r["name"]: r for r in results}[BASELINE_NAME]["folds"]
    rows = []
    for r in results:
        d = r["folds"][metric].values - base[metric].values
        n = len(d)
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "name": r["name"],
            f"d_{metric}": d.mean(),
            "sd_of_diff": sd,
            "se_of_diff": se,
            # mean difference in units of its own standard error; |t| < 2 means
            # indistinguishable from zero on these 25 folds
            "t": d.mean() / se if se and se > 0 else 0.0,
            "folds_better": int((d > 1e-12).sum()),
            "folds_worse": int((d < -1e-12).sum()),
        })
    out = pd.DataFrame(rows).sort_values(f"d_{metric}", ascending=False)
    return out.reset_index(drop=True)


def check_monotone_invariance(df):
    """Show that log1p / rank transforms cannot really help a Random Forest.

    A tree splits on 'feature <= t'. Under a strictly increasing per-feature
    map the training rows fall on the same side of every candidate split, so
    the fitted partition is identical; only the numeric position of the
    threshold BETWEEN two adjacent training values moves (sklearn uses the
    midpoint, and the midpoint of log values is not the log of the midpoint).
    That is enough to flip the occasional test point, which is exactly the
    size of the 'improvement' those variants show.
    """
    sel = selected_cols()
    X = df[sel].values
    Xl = rowwise_log(df, sel).values
    rf_a = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1).fit(X, df[LABEL_COL])
    rf_b = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1).fit(Xl, df[LABEL_COL])
    same_struct = all(
        np.array_equal(a.tree_.feature, b.tree_.feature)
        for a, b in zip(rf_a.estimators_, rf_b.estimators_)
    )
    pa = rf_a.predict_proba(X)[:, 1]
    pb = rf_b.predict_proba(Xl)[:, 1]
    print(f"  โครงสร้างต้นไม้เหมือนกันทุกต้น (raw vs log1p): {same_struct}")
    print(f"  ความต่างของ proba สูงสุดบนข้อมูลเทรน: {np.abs(pa - pb).max():.6f}")
    print("  => log/quantile ไม่ได้เพิ่มข้อมูลให้ RF เลย ต่างกันแค่ตำแหน่งเส้นแบ่ง")


def main():
    df = load_df()
    y = df[LABEL_COL].values
    groups = df[ID_COL].values

    print(f"{len(df)} หัว | {int((y == 1).sum())} พบ / {int((y == 0).sum())} ไม่พบ")

    print("\n=== ตรวจสอบ: RF ไม่สนใจการแปลงแบบ monotone ต่อฟีเจอร์ ===")
    check_monotone_invariance(df)

    print("\n=== ตารางหลัก: ภาพจริงตามที่วัดได้ (as-measured) ===")
    main_table, main_results = run(df, y, groups)

    print("\n" + "=" * 100)
    print("ตารางเปรียบเทียบทุกวิธี (เรียงตาม kappa)")
    print("=" * 100)
    print_table(main_table)

    print("\nค่า SD ระหว่าง fold (ตัวเลขที่ต้องชนะให้ได้ก่อนถึงจะเรียกว่าดีขึ้นจริง):")
    sd = main_table[["name", "accuracy_sd", "kappa_sd", "auc_sd"]].copy()
    for c in sd.columns[1:]:
        sd[c] = sd[c].round(3)
    print(sd.to_string(index=False))

    print("\n" + "=" * 100)
    print("เทียบแบบจับคู่ fold ต่อ fold กับ baseline (25 fold เดียวกันเป๊ะ) — kappa")
    print("=" * 100)
    pk = paired_vs_baseline(main_results, "kappa")
    print(pk.round(3).to_string(index=False))
    print("\nเทียบแบบจับคู่ — accuracy")
    pa = paired_vs_baseline(main_results, "accuracy")
    print(pa.round(3).to_string(index=False))
    print("\n|t| < 2.0 = แยกไม่ออกจาก 0 บน 25 fold นี้ (อยู่ในระดับ noise)")

    # --- exposure stress test -------------------------------------------
    # Same variants, but every head is given its own exposure factor first.
    # A transform that is genuinely exposure-robust should barely move between
    # the two tables; one that leans on absolute brightness should lose
    # ground. Averaged over 3 independent draws so the verdict does not hang
    # on one lucky/unlucky set of factors.
    n_draws, pct = 3, 0.15
    print(f"\n=== ตารางเสริม: จำลองความสว่างเพี้ยน +/-{int(pct*100)}% ต่อหัว "
          f"(เฉลี่ย {n_draws} ครั้งสุ่ม) ===")
    stress_tables = []
    for i in range(n_draws):
        rng = np.random.default_rng(20260729 + i)
        t, _ = run(perturb_exposure(df, rng, pct=pct), y, groups, quiet=True)
        stress_tables.append(t.set_index("name"))
        print(f"  draw {i + 1}/{n_draws} เสร็จ")
    num = [c for c in stress_tables[0].columns if stress_tables[0][c].dtype.kind == "f"]
    stress_table = sum(t[num] for t in stress_tables) / n_draws
    stress_table = stress_table.reset_index().sort_values("kappa", ascending=False)

    print("\n" + "=" * 100)
    print(f"ผลภายใต้ความสว่างเพี้ยน +/-{int(pct*100)}%")
    print("=" * 100)
    print(stress_table[["name", "accuracy", "recall", "specificity", "kappa", "auc"]]
          .round(3).to_string(index=False))

    delta = (stress_table.set_index("name")["kappa"] - main_table.set_index("name")["kappa"])
    print("\nkappa ที่เปลี่ยนไปเมื่อความสว่างเพี้ยน (ติดลบ = แย่ลง):")
    print(delta.sort_values(ascending=False).round(3).to_string())

    main_table.insert(1, "condition", "as-measured")
    stress_table.insert(1, "condition", f"exposure+-{int(pct*100)}%")
    pk.insert(1, "condition", "paired-diff-vs-baseline (kappa)")
    pa.insert(1, "condition", "paired-diff-vs-baseline (accuracy)")
    full = pd.concat([main_table, stress_table, pk, pa], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"\nบันทึกตารางเต็มไว้ที่ {OUT_PATH}  ({len(full)} แถว)")


if __name__ == "__main__":
    main()
