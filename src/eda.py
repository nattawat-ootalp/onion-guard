"""
Phase 2: EDA + baseline.

1. Data quality: missing values, class balance.
2. Univariate AUC ranking across every feature column — which single
   feature separates compactdry=0/1 best, in either direction. This is what
   Phase 3's Random Forest feature importances will later be compared
   against (e.g. to decide whether NDFI is worth fixing/keeping).
3. Distribution plots per feature, colored by class, saved as grid images.
4. Baseline: ROC + AUC for BASELINE_FEATURE, Youden's J cutoff ->
   sensitivity/specificity. This is the bar the Phase 3 model has to beat.

Colors follow the project's validated categorical palette (dataviz skill):
slot 1 blue for compactdry=0, slot 2 orange for compactdry=1 — this pair
already passes the palette's adjacent-pair CVD/contrast gates.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc

from common import ROOT

FEATURES_PATH = ROOT / "data" / "features.csv"
REPORTS_DIR = ROOT / "reports"

LABEL_COL = "compactdry"
ID_COL = "sample_code"

# Baseline single feature to beat with the real model (Phase 3). Chosen
# because it's the cleanest class-separator found during feature review,
# not just the first one that happened to look decent (F_p95, tried
# earlier, is NOT this clean a separator).
BASELINE_FEATURE_BASE = "n_small_sharp_blobs"

NEG_COLOR = "#2a78d6"   # categorical slot 1 (blue) — compactdry=0
POS_COLOR = "#eb6834"   # categorical slot 2 (orange) — compactdry=1
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def check_data_quality(df):
    print("=== Data quality ===")
    n_missing = int(df.isna().sum().sum())
    print(f"Missing values: {n_missing} (across {df.shape[0]} rows x {df.shape[1]} cols)")

    counts = df[LABEL_COL].value_counts().sort_index()
    total = len(df)
    print("Class balance:")
    for label, n in counts.items():
        tag = "compactdry=1 (พบ)" if label == 1 else "compactdry=0 (ไม่พบ)"
        print(f"  {tag}: {n}/{total} ({n/total:.0%})")
    print()


def rank_features_by_auc(df, feature_cols):
    y = df[LABEL_COL].values
    rows = []
    for col in feature_cols:
        score = df[col].values
        if np.all(score == score[0]):
            auc_raw = 0.5
        else:
            auc_raw = roc_auc_score(y, score)
        rows.append({
            "feature": col,
            "auc_raw": auc_raw,
            "auc_effective": max(auc_raw, 1 - auc_raw),
            "direction": "higher -> compactdry=1" if auc_raw >= 0.5 else "lower -> compactdry=1",
        })
    ranked = pd.DataFrame(rows).sort_values("auc_effective", ascending=False).reset_index(drop=True)
    return ranked


def plot_distributions(df, feature_cols, out_path, title):
    n = len(feature_cols)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.4))
    axes = np.atleast_1d(axes).flatten()

    neg = df[df[LABEL_COL] == 0]
    pos = df[df[LABEL_COL] == 1]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        both = df[col].values
        lo, hi = both.min(), both.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 12)
        ax.hist(neg[col], bins=bins, color=NEG_COLOR, alpha=0.6, edgecolor="white", linewidth=0.5, zorder=2)
        ax.hist(pos[col], bins=bins, color=POS_COLOR, alpha=0.6, edgecolor="white", linewidth=0.5, zorder=2)
        short_name = col.replace("_viewmean", "").replace("_viewmax", "")
        ax.set_title(short_name, fontsize=8, color=TEXT_PRIMARY)
        _style_axis(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, alpha=0.6),
    ]
    fig.legend(handles, ["compactdry = 0 (not found)", "compactdry = 1 (found)"],
               loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=12, color=TEXT_PRIMARY, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def baseline_roc(df, feature_col):
    y = df[LABEL_COL].values
    score = df[feature_col].values
    auc_raw = roc_auc_score(y, score)
    flipped = auc_raw < 0.5
    score_for_roc = -score if flipped else score

    fpr, tpr, thresholds = roc_curve(y, score_for_roc)
    auc_val = sk_auc(fpr, tpr)

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]
    best_thresh_internal = thresholds[best_idx]
    # translate back to the original (un-flipped) feature scale for reporting
    best_thresh = -best_thresh_internal if flipped else best_thresh_internal

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=NEG_COLOR, linewidth=2, zorder=3, label=f"ROC ({feature_col})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color=POS_COLOR, s=60, zorder=4,
               label="Youden's J optimal cutoff")
    ax.annotate(
        f"cutoff = {best_thresh:.2f}\nsensitivity = {sensitivity:.2f}\nspecificity = {specificity:.2f}",
        xy=(fpr[best_idx], tpr[best_idx]), xytext=(fpr[best_idx] + 0.08, tpr[best_idx] - 0.22),
        fontsize=9, color=TEXT_PRIMARY,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    ax.set_xlabel("False Positive Rate (1 - specificity)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("True Positive Rate (sensitivity)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"Baseline ROC — {feature_col}\nAUC = {auc_val:.3f}", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _style_axis(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    out_path = REPORTS_DIR / "phase2_baseline_roc.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")

    return {
        "feature": feature_col,
        "auc": auc_val,
        "direction_flipped": flipped,
        "cutoff": best_thresh,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)

    check_data_quality(df)

    feature_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    ranked = rank_features_by_auc(df, feature_cols)
    ranked.to_csv(REPORTS_DIR / "phase2_feature_auc_ranking.csv", index=False)
    print("=== Top 10 single features by effective AUC ===")
    print(ranked.head(10).to_string(index=False))
    print()

    # features.csv carries either plain names (single view) or _viewmean/
    # _viewmax pairs (multiple views) — see extract_features.aggregate_views.
    # Adapt instead of assuming, so this runs under either capture protocol.
    multi_view = any("_viewmean" in c for c in feature_cols)

    if multi_view:
        base_features = sorted({c.replace("_viewmean", "").replace("_viewmax", "") for c in feature_cols})
        plot_distributions(df, [f"{b}_viewmean" for b in base_features],
                            REPORTS_DIR / "phase2_distributions_viewmean.png",
                            "Feature distributions by class (mean-across-views aggregation)")
        plot_distributions(df, [f"{b}_viewmax" for b in base_features],
                            REPORTS_DIR / "phase2_distributions_viewmax.png",
                            "Feature distributions by class (max-across-views aggregation)")
        mean_col, max_col = f"{BASELINE_FEATURE_BASE}_viewmean", f"{BASELINE_FEATURE_BASE}_viewmax"
        mean_auc = ranked.loc[ranked["feature"] == mean_col, "auc_effective"].iloc[0]
        max_auc = ranked.loc[ranked["feature"] == max_col, "auc_effective"].iloc[0]
        chosen_col = mean_col if mean_auc >= max_auc else max_col
        print(f"=== Baseline: {BASELINE_FEATURE_BASE} ===")
        print(f"  {mean_col}: AUC={mean_auc:.3f} | {max_col}: AUC={max_auc:.3f} -> using {chosen_col}")
    else:
        plot_distributions(df, sorted(feature_cols),
                            REPORTS_DIR / "phase2_distributions.png",
                            "Feature distributions by class (single view)")
        chosen_col = BASELINE_FEATURE_BASE
        auc = ranked.loc[ranked["feature"] == chosen_col, "auc_effective"].iloc[0]
        print(f"=== Baseline: {BASELINE_FEATURE_BASE} ===")
        print(f"  {chosen_col}: AUC={auc:.3f} (single view — no mean/max variants)")

    result = baseline_roc(df, chosen_col)
    print(f"\nBaseline result ({result['feature']}):")
    print(f"  AUC = {result['auc']:.3f}")
    print(f"  Youden's J cutoff = {result['cutoff']:.3f}"
          f" ({'score BELOW cutoff -> compactdry=1' if result['direction_flipped'] else 'score ABOVE cutoff -> compactdry=1'})")
    print(f"  Sensitivity = {result['sensitivity']:.3f}, Specificity = {result['specificity']:.3f}")
    print("\nThis is the line Phase 3's Random Forest has to beat.")


if __name__ == "__main__":
    main()
