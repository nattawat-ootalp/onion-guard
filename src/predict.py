"""
Phase 4b: single-head inference — the file that eventually moves to the Pi.

Deliberately minimal imports: numpy, opencv (cv2), joblib. The dev-side
pipeline (extract_features.py / blob_features.py / common.py) uses scipy +
Pillow + pandas, which are fine on a dev machine but heavier than wanted on
a Pi. This file reimplements ONLY the per-head computations needed for one
prediction, using OpenCV equivalents of every scipy operation involved, so
the *numbers* match training even though the *library* differs:

    scipy.ndimage.uniform_filter   -> cv2.boxFilter(normalize=True)
    scipy.ndimage.label            -> cv2.connectedComponentsWithStats(connectivity=4)
    scipy.ndimage.binary_dilation  -> cv2.dilate with a cross kernel (matches
                                       scipy's default cross-shaped structure)
    scipy.ndimage.laplace          -> cv2.Laplacian (same 5-point stencil)
    PIL.Image.open                 -> cv2.imread (+ BGR->RGB conversion)

All thresholds/sizes are read from config.json (plain `json`, no extra
dependency) — nothing is hardcoded twice. The classification cutoff lives
in models/model_config.json specifically because it's the one number most
likely to need retuning once real photos arrive; editing that file alone
is enough, no code changes.

Caveat that this file's import list can't make disappear: joblib.load()
still needs scikit-learn importable at runtime to reconstruct the
RandomForestClassifier object. That's an unavoidable transitive dependency
of loading a pickled sklearn model — the Pi needs scikit-learn installed
even though predict.py itself never imports it directly.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
MODEL_PATH = ROOT / "models" / "model.joblib"
MODEL_CONFIG_PATH = ROOT / "models" / "model_config.json"

CROSS_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Local project module, not a third-party package — importing it does not
# violate this file's "minimal dependencies" contract (still just numpy,
# cv2, joblib at the pip level). It is computed once per head, not per view
# (it compares against a SEPARATE ordinary-light photo), so it is excluded
# from the generic per-view mean/max aggregation below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visible_features as vf_mod  # noqa: E402

EXTRA_HEAD_FEATURE_NAMES = ["uv_exclusive_dot_frac"]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def load_model_config(path=MODEL_CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- ROI / color

def circular_roi_mask(size, center_frac, radius_frac):
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = center_frac[1] * size, center_frac[0] * size
    r = radius_frac * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist <= r


def srgb_to_linear(img_0_1):
    a = 0.055
    return np.where(img_0_1 <= 0.04045, img_0_1 / 12.92, ((img_0_1 + a) / (1 + a)) ** 2.4)


def luminance(img_rgb):
    return (
        img_rgb[..., 0] * 0.2126 + img_rgb[..., 1] * 0.7152 + img_rgb[..., 2] * 0.0722
    ).astype(np.float32)


# ---------------------------------------------------------------- blob detection

def masked_local_baseline(lum, mask_f, size):
    num = cv2.boxFilter(lum * mask_f, ddepth=-1, ksize=(size, size), normalize=True)
    den = cv2.boxFilter(mask_f, ddepth=-1, ksize=(size, size), normalize=True)
    return num / np.maximum(den, 1e-6)


def label_components(binary_mask, min_area, max_area=None):
    mask_u8 = binary_mask.astype(np.uint8)
    n_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
    blobs = []
    kept_mask = np.zeros_like(binary_mask, dtype=bool)
    for i in range(1, n_labels):  # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        cx, cy = centroids[i]
        blobs.append({"area": area, "cx": float(cx), "cy": float(cy), "radius": float(np.sqrt(area / np.pi))})
        kept_mask |= (labels_im == i)
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
    large_blobs, large_mask = label_components(binary_coarse, bd["large_area_min_px"])

    coarse_exclusion = cv2.dilate(
        large_mask.astype(np.uint8), CROSS_KERNEL, iterations=bd["coarse_exclusion_dilate_px"]
    ).astype(bool)
    binary_fine = (residual_fine > thresh_fine) & roi_mask & ~coarse_exclusion
    small_blobs, _ = label_components(binary_fine, bd["min_blob_area_px"], bd["small_area_max_px"])

    return small_blobs, large_blobs


def summarize_blobs(small_blobs, large_blobs):
    n_small, n_large = len(small_blobs), len(large_blobs)
    avg_small_diam = float(np.mean([b["radius"] * 2 for b in small_blobs])) if small_blobs else 0.0
    avg_large_diam = float(np.mean([b["radius"] * 2 for b in large_blobs])) if large_blobs else 0.0

    if n_small >= 2:
        pts = np.array([[b["cx"], b["cy"]] for b in small_blobs])
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        cluster_density = 1.0 / (float(d.min(axis=1).mean()) + 1e-6)
    else:
        cluster_density = 0.0

    ratio_small_to_large = n_small / (n_small + n_large + 1e-6)

    return {
        "n_small_sharp_blobs": n_small,
        "n_large_blotches": n_large,
        "avg_small_blob_diam_px": avg_small_diam,
        "avg_large_blotch_diam_px": avg_large_diam,
        "cluster_density": cluster_density,
        "ratio_small_to_large": ratio_small_to_large,
    }


# ---------------------------------------------------------------- per-view features

def compute_view_features(img_rgb_u8, cfg):
    fe_cfg = cfg["feature_extraction"]
    roi_cfg = cfg["roi"]
    size = img_rgb_u8.shape[0]

    roi_mask = circular_roi_mask(size, roi_cfg["center_fraction"], roi_cfg["radius_fraction"])
    saturated = img_rgb_u8.max(axis=-1) >= fe_cfg["saturation_threshold"]
    valid_mask = roi_mask & ~saturated

    lin = srgb_to_linear(img_rgb_u8 / 255.0)
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

    lap = cv2.Laplacian(brightness.astype(np.float32), cv2.CV_32F, ksize=1)
    v["texture"] = float(lap[valid_mask].std()) if valid_mask.any() else 0.0

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

    small_blobs, large_blobs = detect_blobs(img_rgb_u8.astype(np.float32), roi_mask, cfg)
    all_areas = [b["area"] for b in small_blobs] + [b["area"] for b in large_blobs]
    v["blob_max"] = float(max(all_areas)) if all_areas else 0.0
    v.update(summarize_blobs(small_blobs, large_blobs))

    return v, small_blobs, large_blobs, roi_mask


# ---------------------------------------------------------------- overlay drawing

def draw_overlay(img_bgr, small_blobs, large_blobs):
    """Circles drawn on a COPY of the original (still-BGR, for cv2.imwrite)
    image: yellow for small sharp fungal-dot candidates, magenta for large
    blotch candidates — same colors regardless of which class won, so the
    overlay always shows what was actually detected."""
    out = img_bgr.copy()
    for b in small_blobs:
        center = (int(round(b["cx"])), int(round(b["cy"])))
        radius = max(4, int(round(b["radius"])) + 3)
        cv2.circle(out, center, radius, (0, 255, 255), 2)  # yellow (BGR)
    for b in large_blobs:
        center = (int(round(b["cx"])), int(round(b["cy"])))
        radius = max(6, int(round(b["radius"])) + 5)
        cv2.circle(out, center, radius, (255, 0, 255), 2)  # magenta (BGR)
    return out


# ---------------------------------------------------------------- top-level API

def compute_visible_cross_check(visible_image_path, small_blobs, cfg):
    """Cross-check the UV small blobs against an ALREADY-NORMALIZED (640x640,
    onion-centred) ordinary-light photo of the same head. Returns the
    sentinel (visible_features.sentinel_no_visible) when no photo was
    supplied, or when the detector failed to find the onion in it — a wrong
    alignment used anyway would be worse than marking it unavailable.
    """
    vf_cfg = cfg["visible_features"]
    if not vf_cfg.get("enabled", True) or visible_image_path is None:
        return float(vf_cfg["sentinel_no_visible"])

    img_bgr = cv2.imread(str(visible_image_path))
    if img_bgr is None:
        return float(vf_cfg["sentinel_no_visible"])

    roi_cfg = cfg["roi"]
    roi_mask = circular_roi_mask(img_bgr.shape[0], roi_cfg["center_fraction"], roi_cfg["radius_fraction"])
    mask = vf_mod.discoloration_mask(img_bgr, roi_mask, cfg)
    return vf_mod.uv_exclusive_dot_fraction(small_blobs, mask, cfg)


def predict_head(image_paths, model=None, cfg=None, model_cfg=None, visible_image_path=None):
    """image_paths: UV angle file path(s) for one head, already normalized
    (centred, fixed scale) by the caller — see onion_detect.normalize_to_onion.
    visible_image_path: optional companion ordinary-light photo of the SAME
    head, ALSO already normalized the same way (onion_detect.detect_onion_visible
    + normalize_to_onion) — this file does no onion detection itself, by
    design (see module docstring: framing happens upstream, this file only
    measures an already-framed image).
    Returns {label, confidence, processing_time, overlay_image, overlay_source_path}.
    overlay_image is a BGR numpy array — write it with cv2.imwrite() yourself."""
    start = time.time()
    cfg = cfg or load_config()
    model_cfg = model_cfg or load_model_config()
    if model is None:
        model = joblib.load(MODEL_PATH)

    feature_names = model_cfg["feature_names"]
    # The saved model dictates the naming: a single-view model uses plain
    # feature names, a multi-view one uses _viewmean/_viewmax pairs. Both are
    # built below and the model's own list selects what it needs, so predict
    # stays correct whichever the model was trained as. Cross-modal features
    # (EXTRA_HEAD_FEATURE_NAMES) are excluded here: they don't exist in
    # per-view feats at all, so the generic mean/max loop below would KeyError.
    base_names = sorted({
        f.replace("_viewmean", "").replace("_viewmax", "") for f in feature_names
    } - set(EXTRA_HEAD_FEATURE_NAMES))

    per_view_feats, per_view_blobs, imgs_bgr = [], [], []
    for p in image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        imgs_bgr.append(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        feats, small_blobs, large_blobs, _ = compute_view_features(img_rgb, cfg)
        per_view_feats.append(feats)
        per_view_blobs.append((small_blobs, large_blobs))

    agg = {}
    for b in base_names:
        vals = [pv[b] for pv in per_view_feats]
        agg[b] = float(np.mean(vals))
        agg[f"{b}_viewmean"] = float(np.mean(vals))
        agg[f"{b}_viewmax"] = float(np.max(vals))

    # Computed even when the current model does not use it. Every scan's
    # feature dict is stored and later becomes training data, so gating this
    # on the deployed model's feature list would make the feature impossible
    # to reintroduce: no model asks for it -> it never gets recorded -> no
    # data exists to train a model that asks for it.
    # Only the LAST view's blobs feed the cross-check, matching
    # extract_features.py's training-side convention.
    last_small_blobs = per_view_blobs[-1][0] if per_view_blobs else []
    agg["uv_exclusive_dot_frac"] = compute_visible_cross_check(
        visible_image_path, last_small_blobs, cfg)

    missing = [f for f in feature_names if f not in agg]
    if missing:
        raise ValueError(f"ฟีเจอร์ที่โมเดลต้องการไม่ครบ: {missing[:5]}")
    X = np.array([[agg[f] for f in feature_names]])
    proba_positive = float(model.predict_proba(X)[0, 1])
    threshold = model_cfg["decision_threshold"]
    label = int(proba_positive >= threshold)
    confidence = proba_positive if label == 1 else 1 - proba_positive

    # Overlay the view with the most detected anomalies (most informative angle).
    view_scores = [len(sb) + len(lb) for sb, lb in per_view_blobs]
    overlay_idx = int(np.argmax(view_scores)) if any(view_scores) else 0
    small_blobs, large_blobs = per_view_blobs[overlay_idx]
    overlay_image = draw_overlay(imgs_bgr[overlay_idx], small_blobs, large_blobs)

    return {
        "label": label,
        "confidence": confidence,
        # Raw probability and threshold are exposed because `confidence` alone
        # is misleading in the gap between 0.5 and the threshold: there the
        # model leans positive while the label reads negative, and confidence
        # is reported as 1-proba, which looks like weak evidence FOR "clean".
        "proba_positive": proba_positive,
        "decision_threshold": threshold,
        "borderline": 0.5 <= proba_positive < threshold,
        "processing_time": time.time() - start,
        "overlay_image": overlay_image,
        "overlay_source_path": str(image_paths[overlay_idx]),
        "features": agg,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <view1.png> <view2.png> <view3.png> <view4.png> <out_overlay.png>")
        sys.exit(1)
    result = predict_head(sys.argv[1:5])
    cv2.imwrite(sys.argv[5], result["overlay_image"])
    print(f"label={result['label']} confidence={result['confidence']:.3f} "
          f"processing_time={result['processing_time']:.3f}s "
          f"overlay_source={result['overlay_source_path']} -> {sys.argv[5]}")
