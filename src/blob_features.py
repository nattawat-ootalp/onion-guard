"""
Blob-cluster features that try to tell real fungal fluorescence (many small,
sharp-edged dots, often clustered) apart from ordinary blemishes/stains
(one big, soft-edged patch).

This module is a preview of a small slice of the eventual Phase 1 feature
extraction (src/extract_features.py) — it only covers the dot/cluster
features requested for the mock-data revision, not the full feature set
described in the report (sRGB->linear, mean/SD color, NDFI, etc.).

Approach: a small dot and a large stain differ mainly by SCALE, so detect
them at two different scales rather than trying to score "sharpness" with
one generic metric:
  - FINE baseline: local average over a window bigger than a dot but
    smaller than a stain. Subtracting it isolates high-frequency anomalies
    (dots) while a stain's much slower gradient washes out almost
    completely (the window sees mostly "more stain", so its own local
    average tracks it closely -> near-zero residual).
  - COARSE baseline: local average over a window bigger than the largest
    stain. Subtracting it isolates the stain's overall brightening; a dot
    is far too small to move an average taken over such a wide window, so
    it barely shows up here.
Both baselines use ROI-masked normalized filtering (uniform_filter(lum*mask)
/ uniform_filter(mask)) so the black background just outside the circular
ROI never drags the average down near the edge — a naive unmasked filter
creates a bright false "ring" of anomalies all around the ROI boundary.

Small-scale blobs falling inside a detected coarse (stain) region are
excluded, so a stain's own texture doesn't get double-counted as fungal
dots.

Known limitation (fine for mock validation, worth revisiting on real
photos): if dots in a severe cluster are packed close enough to touch,
connected-component labeling merges them into one blob instead of counting
each separately. The mock generator enforces a minimum dot separation to
keep this rare; real Phase 1 may want a watershed split if it turns out to
matter on real images.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import ROOT, load_config, circular_roi_mask, luminance

IMAGES_DIR = ROOT / "images"
METADATA_PATH = ROOT / "data" / "mock_metadata.csv"
OUT_PATH = ROOT / "data" / "blob_features_debug.csv"

FEATURE_NAMES = [
    "n_small_sharp_blobs",
    "n_large_blotches",
    "avg_small_blob_diam_px",
    "avg_large_blotch_diam_px",
    "cluster_density",
    "ratio_small_to_large",
]


def masked_local_baseline(lum, mask_f, size):
    """Local mean of `lum` over a `size`-wide window, counting only pixels
    inside the mask — avoids background outside the ROI biasing the
    baseline near the ROI's own boundary."""
    num = ndimage.uniform_filter(lum * mask_f, size=size)
    den = ndimage.uniform_filter(mask_f, size=size)
    return num / np.maximum(den, 1e-6)


def _label_components(binary_mask, min_area, max_area=None):
    """Returns (blobs passing the area filter, mask containing ONLY those
    components) — the mask is what should drive any downstream exclusion,
    never the raw pre-filter binary_mask, since a bright small dot can also
    nudge the coarse threshold without being a real large blotch."""
    labeled, n = ndimage.label(binary_mask)
    blobs = []
    kept_mask = np.zeros_like(binary_mask, dtype=bool)
    for i in range(1, n + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        ys, xs = np.where(comp)
        blobs.append({
            "area": area,
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "radius": float(np.sqrt(area / np.pi)),
        })
        kept_mask |= comp
    return blobs, kept_mask


def detect_blobs(img_rgb, roi_mask, cfg):
    bd = cfg["blob_detection"]
    lum = luminance(img_rgb)
    mask_f = roi_mask.astype(np.float32)

    baseline_fine = masked_local_baseline(lum, mask_f, bd["fine_filter_size_px"])
    baseline_coarse = masked_local_baseline(lum, mask_f, bd["coarse_filter_size_px"])
    residual_fine = lum - baseline_fine
    residual_coarse = lum - baseline_coarse

    fine_vals = residual_fine[roi_mask]
    coarse_vals = residual_coarse[roi_mask]
    thresh_fine = max(bd["anomaly_std_k_fine"] * fine_vals.std(), bd["min_abs_threshold_fine"])
    thresh_coarse = max(bd["anomaly_std_k_coarse"] * coarse_vals.std(), bd["min_abs_threshold_coarse"])

    binary_coarse = (residual_coarse > thresh_coarse) & roi_mask
    large_blobs, large_mask = _label_components(binary_coarse, bd["large_area_min_px"])

    coarse_exclusion = ndimage.binary_dilation(
        large_mask, iterations=bd["coarse_exclusion_dilate_px"]
    )
    binary_fine = (residual_fine > thresh_fine) & roi_mask & ~coarse_exclusion
    small_blobs, _ = _label_components(binary_fine, bd["min_blob_area_px"], bd["small_area_max_px"])

    return small_blobs, large_blobs


def summarize_blobs(small_blobs, large_blobs):
    n_small, n_large = len(small_blobs), len(large_blobs)
    avg_small_diam = float(np.mean([b["radius"] * 2 for b in small_blobs])) if small_blobs else 0.0
    avg_large_diam = float(np.mean([b["radius"] * 2 for b in large_blobs])) if large_blobs else 0.0

    if n_small >= 2:
        pts = np.array([[b["cx"], b["cy"]] for b in small_blobs])
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mean_nn_dist = float(d.min(axis=1).mean())
        cluster_density = 1.0 / (mean_nn_dist + 1e-6)
    else:
        cluster_density = 0.0

    # Bounded 0-1 fraction rather than a raw n_small/n_large ratio, which
    # would blow up to a meaningless huge number whenever n_large is 0
    # (the common case for genuinely fungal-only heads).
    ratio_small_to_large = n_small / (n_small + n_large + 1e-6)

    return {
        "n_small_sharp_blobs": n_small,
        "n_large_blotches": n_large,
        "avg_small_blob_diam_px": avg_small_diam,
        "avg_large_blotch_diam_px": avg_large_diam,
        "cluster_density": cluster_density,
        "ratio_small_to_large": ratio_small_to_large,
    }


def extract_blob_features(img_rgb, cfg):
    """Single entry point later reused by extract_features.py: RGB array in, feature dict out."""
    roi_cfg = cfg["roi"]
    size = img_rgb.shape[0]
    roi_mask = circular_roi_mask(size, roi_cfg["center_fraction"], roi_cfg["radius_fraction"])
    small_blobs, large_blobs = detect_blobs(img_rgb, roi_mask, cfg)
    return summarize_blobs(small_blobs, large_blobs)


def main():
    cfg = load_config()
    meta_df = pd.read_csv(METADATA_PATH)
    n_views = cfg["samples"]["n_views"]

    rows = []
    for _, row in meta_df.iterrows():
        code = row["sample_code"]
        per_view = []
        for v in range(1, n_views + 1):
            img = np.array(Image.open(IMAGES_DIR / f"{code}_V{v}.png").convert("RGB"))
            per_view.append(extract_blob_features(img, cfg))

        agg = {"sample_code": code, "compactdry": row["compactdry"],
               "severity": row["severity"], "hard_negative": row["hard_negative"]}
        for feat in FEATURE_NAMES:
            vals = [pv[feat] for pv in per_view]
            agg[f"{feat}_mean"] = float(np.mean(vals))
            agg[f"{feat}_max"] = float(np.max(vals))
        rows.append(agg)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(out_df)} rows)")

    print("\nGroup means (mean-across-views features) — sanity check that the four "
          "features separate mild/severe/hard-negative/clean:")
    def group_label(r):
        if r["compactdry"] == 1:
            return f"positive_{r['severity']}"
        return "negative_hard" if r["hard_negative"] else "negative_clean"

    out_df["group"] = out_df.apply(group_label, axis=1)
    cols = [f"{f}_mean" for f in FEATURE_NAMES]
    print(out_df.groupby("group")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
