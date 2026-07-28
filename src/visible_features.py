"""
Cross-check UV-detected dots against an ordinary-light photo of the same head.

WHY THIS EXISTS
The report's own premise is that real fungal fluorescence is "not yet
visible to the naked eye". That is directly testable: if a dot the UV
detector found also shows up as ordinary surface discoloration under normal
light (dry skin, a bruise, dirt), it is weaker evidence of fluorescence and
stronger evidence of an everyday blemish. Measured on one real photo pair,
dry/tan-coloured skin patches covered 17% of the visible-light surface but
accounted for 30% of the UV blobs' locations — a real, non-trivial overlap,
though most UV blobs (70%) did NOT land on visible discoloration.

CALIBRATION CAVEAT
The hue range and saturation threshold below were tuned on exactly ONE real
photo pair. They are a reasonable starting point, not a validated cutoff —
revisit once more paired photos are available and check whether the 17%/30%
split holds.

USAGE CONTRACT
Both images must already be independently normalized through
onion_detect.normalize_to_onion (same onion_radius_frac) before calling
anything here. See onion_detect.detect_onion_visible's docstring for why
that alone is enough alignment without an explicit registration step.

Kept to cv2 + numpy only (like onion_detect.py) so predict.py can import it
without breaking its own minimal-dependency contract.
"""
import cv2
import numpy as np


def discoloration_mask(visible_bgr_normalized, roi_mask, cfg):
    """Boolean mask of dry/tan-coloured surface discoloration within the ROI.

    Hue range and saturation floor come from config visible_features —
    tuned against real dry shallot skin, which reads as yellow/tan (OpenCV
    hue ~10-40) against the surrounding pink/purple skin (hue ~150-180).
    """
    vf = cfg["visible_features"]
    hsv = cv2.cvtColor(visible_bgr_normalized, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    lo, hi = vf["discoloration_hue_range"]
    mask = (hue >= lo) & (hue < hi) & (sat >= vf["discoloration_sat_min"])
    return mask & roi_mask


def uv_exclusive_dot_fraction(small_blobs, mask, cfg):
    """Fraction of small_sharp_blobs whose centre does NOT land on a
    discoloration patch — i.e. how much of the UV signal is unexplained by
    anything visible in ordinary light. Higher = more of the detected dots
    look like genuine fluorescence rather than ordinary surface blemish,
    which keeps the direction consistent with every other blob feature
    (higher = more fungal-like).

    Returns cfg's sentinel_no_visible when there is nothing to check against
    (mask is None, e.g. no visible-light photo was supplied) — NOT when there
    are simply zero blobs, which is a real, informative "nothing suspicious
    found" case and returns 0.0.
    """
    vf = cfg["visible_features"]
    if mask is None:
        return float(vf["sentinel_no_visible"])
    if not small_blobs:
        return 0.0

    h, w = mask.shape
    on_discoloration = 0
    for b in small_blobs:
        y = min(max(int(round(b["cy"])), 0), h - 1)
        x = min(max(int(round(b["cx"])), 0), w - 1)
        if mask[y, x]:
            on_discoloration += 1
    frac_on = on_discoloration / len(small_blobs)
    return 1.0 - frac_on
