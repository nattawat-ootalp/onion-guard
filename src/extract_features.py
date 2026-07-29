"""
Phase 1 feature extraction: turns each head's 4 UV photos into one feature
row in data/features.csv.

Covers the report's base feature set (mean/SD of R,G,B; mean/SD brightness;
texture; too-bright/too-dim area anomalies) plus the extra detail features
requested alongside it (F_p95/F_p05, A_high/A_low via a robust per-head
threshold, blob_max, NDFI) and the dot-cluster features built for the
mock-data revision (src/blob_features.py) — small-sharp-dot count/density/
size vs large-blotch count/size, which is what actually tells a real
fungal cluster apart from an ordinary stain.

FEATURE_NAMES below is the single place to add or remove a feature: every
name in it must be a key returned by compute_view_features(), and gets
aggregated (mean AND max across the head's 4 views, since the book wants
the average look but a defect can show up strongly from just one angle)
into two columns in the final CSV.

NDFI ("Normalized Difference Fluorescence Index") is my own working
definition, not a formula pulled from the report — I don't have its exact
wording. Physical premise: healthy tissue fluoresces purple/reddish (G much
lower than R and B), degrading tissue fluoresces bright white (R, G, B
converge). So NDFI = (G - avg(R,B)) / (G + avg(R,B)) rises from clearly
negative (background) toward less-negative/positive (defect). Flag this
for correction once you have the report's actual definition in front of
you — everything downstream just treats it as one more feature column, so
swapping the formula later is a one-function change.

Saturated pixels (any channel >= feature_extraction.saturation_threshold)
are excluded from the color/brightness/texture statistics — a clipped
pixel doesn't carry real brightness information and would bias the mean/SD
— but they're kept in the anomaly-area and blob-detection passes, since a
blown-out pixel IS itself a legitimate "too bright" anomaly.
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import ROOT, load_config, circular_roi_mask, srgb_to_linear
from blob_features import detect_blobs, summarize_blobs
from onion_detect import detect_onion, detect_onion_visible, normalize_to_onion
import visible_features as vf_mod

IMAGES_DIR = ROOT / "images"
LABELS_PATH = ROOT / "data" / "labels.csv"
OUT_PATH = ROOT / "data" / "features.csv"

FEATURE_NAMES = [
    # ตามเล่ม: mean/SD ของสี + ความสว่างเฉลี่ย/SD + พื้นผิว
    "mean_R", "mean_G", "mean_B",
    "sd_R", "sd_G", "sd_B",
    "brightness_mean", "brightness_sd",
    "texture",
    # ตามเล่ม: พื้นที่ผิดปกติ (สว่างเกิน / หรี่เกิน จากค่ากลางของหัวนั้น แยกกัน)
    "A_high", "A_low",
    # เพิ่มเติม: ปลายการกระจายทั้งสองด้าน + จุดเด่นที่สุด + NDFI
    "F_p95", "F_p05",
    "blob_max",
    "NDFI_mean", "NDFI_p95", "NDFI_p05",
    # ฟีเจอร์กลุ่มจุด/ปื้น จาก blob_features.py (แยกจุดเชื้อเล็กคมชัดจากคราบใหญ่เบลอ)
    "n_small_sharp_blobs", "n_large_blotches",
    "avg_small_blob_diam_px", "avg_large_blotch_diam_px",
    "cluster_density", "ratio_small_to_large",
]

# Computed ONCE per head (not per view, not mean/max'd) because it compares
# against a SEPARATE photo (ordinary light) rather than describing the UV
# photo itself. Value is visible_features.sentinel_no_visible when no
# "{code}_VISIBLE.png" companion file exists — real for every mock sample
# today, so this column is currently constant; it becomes informative once
# real paired photos start arriving. See src/visible_features.py.
EXTRA_HEAD_FEATURES = ["uv_exclusive_dot_frac"]


def preprocess_image(img_bgr, cfg):
    """Apply the SAME framing the serving path applies before measuring.

    web/app.py re-frames every uploaded photo with onion_detect so the onion
    is centred at a fixed scale. If training features were computed from raw
    images instead, the model would be fitted on differently-processed inputs
    than it later scores — measured on one mock image, that skew moved sd_B
    from 0.023 to 0.036 and texture from 0.030 to 0.023. Same code, both
    sides, so the numbers mean the same thing.

    Set feature_extraction.apply_auto_framing to false only if the images are
    known to be pre-framed, and remember the serving path must match.
    """
    if not cfg["feature_extraction"].get("apply_auto_framing", True):
        return img_bgr

    od = cfg["onion_detect"]
    sanity = od.get("sanity", {})
    detection = detect_onion(
        img_bgr, k=od["detect_k"], min_area_frac=od["min_area_frac"],
        min_radius_frac=sanity.get("min_radius_frac_of_frame"),
        max_radius_frac=sanity.get("max_radius_frac_of_frame"),
    )
    framed, _ = normalize_to_onion(img_bgr, cfg["image"]["size_px"],
                                    od["onion_radius_frac"], detection=detection)
    return framed


def aggregate_views(per_view):
    """Combine per-view features into the one row that represents a head.

    With several angles each feature becomes a mean/max pair — the mean is
    the head's overall look, the max catches a defect visible from only one
    angle. With a SINGLE angle those two are identical by definition, so the
    pair is dropped and the plain name is used: duplicated columns add no
    information and split a Random Forest's importance between two identical
    features, making the ranking harder to read.

    This is why the feature names depend on the view count. Changing
    samples.n_views therefore requires re-extracting features and retraining.
    """
    if len(per_view) == 1:
        return {feat: float(per_view[0][feat]) for feat in FEATURE_NAMES}

    agg = {}
    for feat in FEATURE_NAMES:
        vals = [pv[feat] for pv in per_view]
        agg[f"{feat}_viewmean"] = float(np.mean(vals))
        agg[f"{feat}_viewmax"] = float(np.max(vals))
    return agg


def compute_view_features(img_u8, cfg):
    """img_u8: original 0-255 sRGB array straight off disk."""
    fe_cfg = cfg["feature_extraction"]
    roi_cfg = cfg["roi"]
    size = img_u8.shape[0]

    roi_mask = circular_roi_mask(size, roi_cfg["center_fraction"], roi_cfg["radius_fraction"])
    saturated = img_u8.max(axis=-1) >= fe_cfg["saturation_threshold"]
    valid_mask = roi_mask & ~saturated

    lin = srgb_to_linear(img_u8 / 255.0)  # sRGB -> linear light, 0-1 scale
    R, G, B = lin[..., 0], lin[..., 1], lin[..., 2]
    brightness = (R + G + B) / 3.0

    v = {}
    for name, chan in (("R", R), ("G", G), ("B", B)):
        vals = chan[valid_mask]
        v[f"mean_{name}"] = float(vals.mean()) if vals.size else 0.0
        v[f"sd_{name}"] = float(vals.std()) if vals.size else 0.0

    b_valid = brightness[valid_mask]
    v["brightness_mean"] = float(b_valid.mean()) if b_valid.size else 0.0
    v["brightness_sd"] = float(b_valid.std()) if b_valid.size else 0.0

    # Texture: local roughness via Laplacian response on brightness — a
    # smooth healthy surface has low texture energy, a pitted/degraded one
    # has more.
    lap = ndimage.laplace(brightness)
    v["texture"] = float(lap[valid_mask].std()) if valid_mask.any() else 0.0

    # Anomaly area: fraction of the ROI far above/below THIS head's own
    # robust center value (median + MAD, robust to the anomaly pixels
    # themselves skewing the estimate). Uses the full roi_mask (not just
    # valid_mask) since a saturated pixel is itself a "too bright" anomaly.
    med = float(np.median(b_valid)) if b_valid.size else 0.0
    mad = float(np.median(np.abs(b_valid - med))) if b_valid.size else 0.0
    robust_std = 1.4826 * mad
    k = fe_cfg["area_anomaly_mad_k"]
    high_thresh, low_thresh = med + k * robust_std, med - k * robust_std

    roi_brightness = brightness[roi_mask]
    v["A_high"] = float((roi_brightness > high_thresh).mean()) if roi_brightness.size else 0.0
    v["A_low"] = float((roi_brightness < low_thresh).mean()) if roi_brightness.size else 0.0
    v["F_p95"] = float(np.percentile(roi_brightness, 95)) if roi_brightness.size else 0.0
    v["F_p05"] = float(np.percentile(roi_brightness, 5)) if roi_brightness.size else 0.0

    eps = fe_cfg["ndfi_eps"]
    ndfi = (G - (R + B) / 2.0) / (G + (R + B) / 2.0 + eps)
    roi_ndfi = ndfi[roi_mask]
    v["NDFI_mean"] = float(roi_ndfi.mean()) if roi_ndfi.size else 0.0
    v["NDFI_p95"] = float(np.percentile(roi_ndfi, 95)) if roi_ndfi.size else 0.0
    v["NDFI_p05"] = float(np.percentile(roi_ndfi, 5)) if roi_ndfi.size else 0.0

    # Reused dual-scale detector from blob_features.py: blob_max answers
    # "how big is the single most prominent bright spot", the cluster
    # features answer "is this a real fungal dot-cluster or a stain".
    small_blobs, large_blobs = detect_blobs(img_u8, roi_mask, cfg)
    all_areas = [b["area"] for b in small_blobs] + [b["area"] for b in large_blobs]
    v["blob_max"] = float(max(all_areas)) if all_areas else 0.0
    v.update(summarize_blobs(small_blobs, large_blobs))

    return v, small_blobs, large_blobs


def compute_uv_exclusive_feature(code, last_view_blobs, cfg):
    """Look for "{code}_VISIBLE.png" next to the UV images; if present, cross
    -check the last view's small blobs against it. Absent for every mock
    sample today (mock never generated a visible-light companion), so this
    returns the sentinel for all of them — expected, not a bug."""
    vf_cfg = cfg["visible_features"]
    if not vf_cfg.get("enabled", True):
        return float(vf_cfg["sentinel_no_visible"])

    visible_path = IMAGES_DIR / f"{code}_VISIBLE.png"
    if not visible_path.exists():
        return float(vf_cfg["sentinel_no_visible"])

    rgb = np.array(Image.open(visible_path).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    od = cfg["onion_detect"]
    sanity = od.get("sanity", {})
    detection = detect_onion_visible(
        bgr, k=od["visible_light_detect_k"], min_area_frac=od["min_area_frac"],
        min_radius_frac=sanity.get("min_radius_frac_of_frame"),
        max_radius_frac=sanity.get("max_radius_frac_of_frame"),
        min_abs_threshold=od.get("visible_light_min_abs_a", 6.0),
    )
    if not detection[3].get("ok"):
        # Detector failed on this photo — using it anyway would risk a
        # silently wrong alignment, worse than just marking it unavailable.
        return float(vf_cfg["sentinel_no_visible"])

    framed, _ = normalize_to_onion(bgr, cfg["image"]["size_px"], od["onion_radius_frac"],
                                    detection=detection)
    roi_mask = circular_roi_mask(framed.shape[0], cfg["roi"]["center_fraction"],
                                  cfg["roi"]["radius_fraction"])
    mask = vf_mod.discoloration_mask(framed, roi_mask, cfg)
    return vf_mod.uv_exclusive_dot_fraction(last_view_blobs, mask, cfg)


def main():
    cfg = load_config()
    labels_df = pd.read_csv(LABELS_PATH)
    n_views = cfg["samples"]["n_views"]

    rows = []
    n_with_visible = 0
    for _, row in labels_df.iterrows():
        code = row["sample_code"]
        per_view = []
        last_view_blobs = []
        for view in range(1, n_views + 1):
            rgb = np.array(Image.open(IMAGES_DIR / f"{code}_V{view}.png").convert("RGB"))
            # Framing works in BGR (OpenCV convention) like the serving path,
            # then hand the result back as RGB for the feature maths.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            framed = preprocess_image(bgr, cfg)
            img_u8 = cv2.cvtColor(framed, cv2.COLOR_BGR2RGB).astype(np.float32)
            feats, small_blobs, _ = compute_view_features(img_u8, cfg)
            per_view.append(feats)
            last_view_blobs = small_blobs  # only the final view feeds the visible cross-check

        agg = {"sample_code": code}
        agg.update(aggregate_views(per_view))
        uv_excl = compute_uv_exclusive_feature(code, last_view_blobs, cfg)
        if uv_excl != cfg["visible_features"]["sentinel_no_visible"]:
            n_with_visible += 1
        agg["uv_exclusive_dot_frac"] = uv_excl
        rows.append(agg)

    features_df = pd.DataFrame(rows)
    out_df = features_df.merge(labels_df, on="sample_code", how="left")
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(out_df)} rows x {len(out_df.columns)} columns")
    if n_views == 1:
        print(f"({len(FEATURE_NAMES)} features, single view — no mean/max pair "
              f"+ {len(EXTRA_HEAD_FEATURES)} cross-modal + sample_code + compactdry)")
    else:
        print(f"({len(FEATURE_NAMES)} features x {{mean, max}} across {n_views} views "
              f"+ {len(EXTRA_HEAD_FEATURES)} cross-modal + sample_code + compactdry)")
    print(f"Heads with a visible-light companion photo: {n_with_visible}/{len(labels_df)}")


if __name__ == "__main__":
    main()
