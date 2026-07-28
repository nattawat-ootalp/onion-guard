"""Shared helpers used across the mock generator, feature extraction, and
(later) predict.py — kept in one place so ROI geometry / config loading stay
consistent everywhere they're used."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def circular_roi_mask(size, center_frac, radius_frac):
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = center_frac[1] * size, center_frac[0] * size
    r = radius_frac * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist <= r


def luminance(img_rgb):
    return (
        img_rgb[..., 0] * 0.2126 + img_rgb[..., 1] * 0.7152 + img_rgb[..., 2] * 0.0722
    ).astype(np.float32)


def srgb_to_linear(img_srgb_0_1):
    """Standard sRGB EOTF (gamma decode). Input/output in [0, 1]."""
    a = 0.055
    return np.where(
        img_srgb_0_1 <= 0.04045,
        img_srgb_0_1 / 12.92,
        ((img_srgb_0_1 + a) / (1 + a)) ** 2.4,
    )
