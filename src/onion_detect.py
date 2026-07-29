"""
Locate the shallot in a UV frame, then normalize framing and scale.

WHY THIS EXISTS
Phone shots do not land the onion in the same place or at the same size
twice. Measured on the first real Android photo, the onion sat at frame
fraction (0.695, 0.462) instead of the assumed (0.5, 0.5): 31% of it fell
outside the fixed circular ROI, and 84% of that ROI was empty background —
which also let background speckle be counted as anomalies.

Scale matters just as much. The blob detector's thresholds
(min_blob_area_px, small_area_max_px, large_area_min_px) are absolute pixel
counts, so an onion photographed closer or further away changes what
"small dot" and "large patch" mean. Real dots measured ~7px across versus
~13-16px in the mock data, which is exactly this problem.

The fix is to make the framing assumption TRUE rather than to loosen it:
detect the onion, then crop and resize so it is always centred and always
occupies the same fraction of the output. After that the fixed ROI in
config.json is correct by construction, and every pixel threshold means
the same physical size in every photo — no per-image ROI has to be plumbed
through the feature code.

BACKGROUND ASSUMPTION
Under 365nm the box background is blue-dominant haze with almost no red or
green (measured: B=51, R=7, G=6), while the onion carries red and its
fluorescing spots are bright in all channels. Detection therefore looks for
pixels whose red/green rise clearly above the border background rather than
using a fixed threshold, so it adapts to how bright the lamp is on the day.
"""
import cv2
import numpy as np


def _background_stats(img_bgr, border_frac=0.06):
    """Estimate background from the frame border, where the onion should not be."""
    h, w = img_bgr.shape[:2]
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    border = np.concatenate([
        img_bgr[:bh].reshape(-1, 3),
        img_bgr[-bh:].reshape(-1, 3),
        img_bgr[:, :bw].reshape(-1, 3),
        img_bgr[:, -bw:].reshape(-1, 3),
    ])
    return border.mean(axis=0), border.std(axis=0)


def _largest_component_to_detection(mask, h, w, min_area_frac, min_radius_frac, max_radius_frac, info):
    """Shared finish for both detectors: pick the largest blob, validate its
    size, build the (mask, centre, radius, info) tuple normalize_to_onion
    expects. Splitting this out means the UV and visible-light detectors only
    differ in how `mask` gets built, not in how it gets validated."""
    scale = max(3, int(round(min(h, w) * 0.012)) | 1)  # odd kernel, scales with frame
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((scale * 2 + 1,) * 2, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((scale,) * 2, np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        info.update({"ok": False, "reason": "ไม่พบวัตถุที่แยกจากพื้นหลังได้"})
        return mask.astype(bool), (w / 2, h / 2), min(h, w) * 0.4, info

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area_frac * h * w:
        info.update({"ok": False, "reason": f"วัตถุที่เจอเล็กเกินไป ({area} px)",
                     "area_frac": area / (h * w)})
        return mask.astype(bool), (w / 2, h / 2), min(h, w) * 0.4, info

    x, y, bw_, bh_ = (int(stats[idx, c]) for c in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    obj = (labels == idx)
    # Fill interior gaps: dark folds inside the onion are still the onion.
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((scale * 3,) * 2, np.uint8)).astype(bool)

    cx, cy = x + bw_ / 2.0, y + bh_ / 2.0
    radius = max(bw_, bh_) / 2.0

    touches = [n for n, c in (("ซ้าย", x <= 1), ("บน", y <= 1),
                              ("ขวา", x + bw_ >= w - 1), ("ล่าง", y + bh_ >= h - 1)) if c]
    radius_frac = radius / (min(h, w) / 2.0)
    info.update({
        "ok": True,
        "bbox": [x, y, bw_, bh_],
        "area": area,
        "area_frac": area / (h * w),
        "touches_edge": touches,
        "radius_frac_of_frame": radius_frac,
    })

    # Implausible size means the segmenter grabbed the wrong thing. Fall back
    # to a centred crop and say so, rather than re-framing every measurement
    # around a mistake.
    if min_radius_frac is not None and radius_frac < min_radius_frac:
        info.update({"ok": False,
                     "reason": f"วัตถุที่เจอเล็กผิดปกติ (รัศมี {radius_frac:.2f} < {min_radius_frac})"})
        return obj, (w / 2, h / 2), min(h, w) * 0.4, info
    if max_radius_frac is not None and radius_frac > max_radius_frac:
        info.update({"ok": False,
                     "reason": f"วัตถุที่เจอใหญ่ผิดปกติ (รัศมี {radius_frac:.2f} > {max_radius_frac}) "
                               "อาจไปจับพื้นหลังที่สว่าง"})
        return obj, (w / 2, h / 2), min(h, w) * 0.4, info

    return obj, (cx, cy), radius, info


def detect_onion(img_bgr, k=4.0, min_area_frac=0.005,
                  min_radius_frac=None, max_radius_frac=None):
    """Return (mask, (cx, cy), radius, info) for a UV-lit frame.

    radius is half the longer side of the object's bounding box, so an onion
    with a papery tail is still fully enclosed. info carries diagnostics and
    an `ok` flag; callers must handle ok=False rather than trusting the
    geometry, because a failed detection would silently mis-crop every
    downstream measurement.

    min_radius_frac / max_radius_frac bound the plausible onion size as a
    fraction of the half-frame. A detection outside them means the segmenter
    latched onto something else (a speck, or the whole lit background), so it
    is reported as a failure and the caller falls back to a centred crop.
    """
    h, w = img_bgr.shape[:2]
    r, g = img_bgr[:, :, 2].astype(np.float32), img_bgr[:, :, 1].astype(np.float32)
    bg_mean, bg_std = _background_stats(img_bgr)

    # Background is blue-only, so the onion announces itself in red/green.
    thr_r = bg_mean[2] + k * max(bg_std[2], 1.0)
    thr_g = bg_mean[1] + k * max(bg_std[1], 1.0)
    mask = ((r > thr_r) | (g > thr_g)).astype(np.uint8)

    info = {"mode": "uv", "threshold_r": float(thr_r), "threshold_g": float(thr_g),
            "bg_mean_bgr": [float(v) for v in bg_mean]}
    return _largest_component_to_detection(mask, h, w, min_area_frac,
                                            min_radius_frac, max_radius_frac, info)


def detect_onion_visible(img_bgr, k=4.0, min_area_frac=0.005,
                          min_radius_frac=None, max_radius_frac=None,
                          min_abs_threshold=6.0):
    """Return (mask, (cx, cy), radius, info) for the SAME head under ordinary
    room light — the paired photo used to tell true UV-only fluorescence
    apart from ordinary surface discoloration (dry skin, dirt) that would
    show up under any light.

    Keys off the CIELAB a* axis (green -> red), not brightness. The first
    version thresholded HSV value on the assumption that the onion is the
    brightest thing in an otherwise dark frame; measured on the real
    photo set that assumption is simply false — border gray 52 vs onion
    gray 59 on S001 — and it failed on 32 of 60 heads, each time grabbing
    the whole frame. What does separate them is colour: a shallot is red
    and every background used so far (mulberry paper, cloth) is not.
    Measured across all 60 visible-light photos the onion sits a median of
    8.6 background standard deviations up the a* axis.

    a* is chosen over a raw R-B difference because it is defined to be
    independent of lightness, so the same threshold holds whether the room
    light was bright or dim — which matters here precisely because these
    photos are not exposure-controlled the way the UV ones are.

    min_abs_threshold is a floor in a* units. Background a* is extremely
    uniform on plain paper (std ~1.5), so k * std alone can put the
    threshold inside the background's own noise; the floor keeps a nearly
    flat background from producing a threshold that admits half the frame.

    IMPORTANT: this function does NOT try to align with the UV photo via any
    explicit transform. Both photos are independently normalized through
    normalize_to_onion with the same onion_radius_frac, which lines them up
    well enough (measured across the 60 real pairs: median 4% difference in
    detected radius) because each photo is centred and scaled around its OWN
    detected onion. A photo where this detector fails (info['ok'] is False)
    should not be used for the cross-modal comparison, since a fallback
    centred crop from the wrong scale would misalign silently rather than
    obviously.
    """
    h, w = img_bgr.shape[:2]
    a = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)

    bh, bw = max(1, int(h * 0.06)), max(1, int(w * 0.06))
    border = np.concatenate([a[:bh].ravel(), a[-bh:].ravel(),
                             a[:, :bw].ravel(), a[:, -bw:].ravel()])
    bg_a_mean, bg_a_std = float(border.mean()), float(border.std())

    thr_a = max(bg_a_mean + k * bg_a_std, bg_a_mean + min_abs_threshold)
    mask = (a > thr_a).astype(np.uint8)

    info = {"mode": "visible", "threshold_a": float(thr_a),
            "bg_a_mean": bg_a_mean, "bg_a_std": bg_a_std}
    return _largest_component_to_detection(mask, h, w, min_area_frac,
                                            min_radius_frac, max_radius_frac, info)


def normalize_to_onion(img_bgr, target_size, onion_radius_frac, detection=None):
    """Crop around the onion and resize so it is centred at a fixed scale.

    onion_radius_frac is the radius the onion should occupy in the output, as
    a fraction of target_size. Everything downstream then sees the onion in
    the same place at the same size in every photo.
    """
    if detection is None:
        detection = detect_onion(img_bgr)
    _, (cx, cy), radius, info = detection

    # Crop side S must satisfy  radius * (target_size / S) == onion_radius_frac * target_size,
    # i.e. S = radius / onion_radius_frac. half is S/2.
    half = radius / (2.0 * onion_radius_frac)
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(half * 2))

    # Pad rather than clamp: clamping would shift the onion off-centre and
    # silently undo the normalization this function exists to provide.
    h, w = img_bgr.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - w), max(0, y0 + side - h)
    if pad_l or pad_t or pad_r or pad_b:
        img_bgr = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l
        y0 += pad_t

    crop = img_bgr[y0:y0 + side, x0:x0 + side]
    if crop.size == 0:
        crop = img_bgr
    interp = cv2.INTER_AREA if crop.shape[0] > target_size else cv2.INTER_LINEAR
    out = cv2.resize(crop, (target_size, target_size), interpolation=interp)

    info = dict(info)
    info["padded"] = bool(pad_l or pad_t or pad_r or pad_b)
    info["crop_side_px"] = side
    return out, info
