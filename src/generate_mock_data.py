"""
Generate synthetic UV-365nm shallot images + labels.csv for pipeline development.

Real photos aren't available yet (capture box unfinished, cultures still
incubating), so this script fabricates a dataset with the same shape as the
real one will have: N_SAMPLES heads x N_VIEWS angles, one label per head.

Positive heads (compactdry=1) get small SHARP-edged bright dots that cluster
together — that's what real UV fluorescence at degrading tissue looks like.
Mild cases get a few scattered dots; severe cases get many dots packed into
dense clusters. A subset of NEGATIVE heads get a large BLURRY stain/bruise
blotch instead — a non-fungal distractor, deliberately soft-edged and large
so downstream features have to actually separate real dot-clusters from
ordinary blemishes rather than just detecting "anything bright".

All parameters live in config.json so this is easy to retune or swap out
once real images arrive. data/mock_metadata.csv additionally records which
synthetic category (mild/severe/hard-negative/clean) each head got — this is
debug-only scaffolding for validating the mock + features, NOT part of the
real data schema (real labels.csv only ever has sample_code + compactdry).
"""
import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, load_config

IMAGES_DIR = ROOT / "images"
LABELS_PATH = ROOT / "data" / "labels.csv"
METADATA_PATH = ROOT / "data" / "mock_metadata.csv"


def make_labels(cfg, rng):
    s = cfg["samples"]
    n = s["n_samples"]
    assert s["n_positive"] + s["n_negative"] == n, "n_positive + n_negative must equal n_samples"

    labels = np.array([1] * s["n_positive"] + [0] * s["n_negative"])
    rng.shuffle(labels)

    codes = [f"{s['sample_code_prefix']}{i+1:0{s['sample_code_digits']}d}" for i in range(n)]
    df = pd.DataFrame({"sample_code": codes, "compactdry": labels})
    return df


def assign_categories(labels_df, cfg, rng):
    """Roll a synthetic sub-category ONCE per head (not per view): positives
    get mild/severe fungal-dot severity, negatives get a hard-negative stain
    or stay clean. This mirrors reality where severity/blemishes are a
    property of the physical head, not of the camera angle."""
    dots_cfg = cfg["mock"]["fungal_dots"]
    blotch_cfg = cfg["mock"]["hard_negative_blotch"]

    severities = []
    hard_negatives = []
    for _, row in labels_df.iterrows():
        if row["compactdry"] == 1:
            severe = rng.random() < dots_cfg["severity_severe_probability"]
            severities.append("severe" if severe else "mild")
            hard_negatives.append(False)
        else:
            severities.append("none")
            hard_negatives.append(rng.random() < blotch_cfg["probability"])

    meta = labels_df.copy()
    meta["severity"] = severities
    meta["hard_negative"] = hard_negatives
    return meta


def circular_roi_mask_and_dist(size, center_frac, radius_frac):
    """Like common.circular_roi_mask, but also returns the distance-from-center
    array — needed here for the center-bright/edge-dim falloff effect."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = center_frac[1] * size, center_frac[0] * size
    r = radius_frac * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist <= r, dist


def sample_cluster_positions(rng, center, spread_px, n, min_sep_px, max_field_radius_px):
    """Place n points around `center`, jittered by spread_px, rejecting any
    point closer than min_sep_px to an existing one so dots stay distinct
    (not merged into a blob) and closer than max_field_radius_px to center
    so they don't spill outside the onion ROI."""
    positions = []
    attempts = 0
    max_attempts = max(200, n * 60)
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0, 2 * np.pi)
        r = abs(rng.normal(0, spread_px))
        if r > max_field_radius_px:
            continue
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        if all(np.hypot(x - px, y - py) >= min_sep_px for px, py in positions):
            positions.append((x, y))
    return positions


def draw_sharp_dot(img, size, bx, by, radius_px, edge_px, color, intensity):
    """Small bright dot with a crisp edge: full intensity inside radius_px,
    a narrow (edge_px wide) linear falloff, then nothing — the opposite of a
    Gaussian blob whose brightness fades gradually over a wide area."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
    alpha = np.clip((radius_px + edge_px / 2 - dist) / edge_px, 0.0, 1.0)
    blend = alpha[..., None] * (intensity / 255.0) * np.array(color, dtype=np.float32)
    img += blend


def draw_blurry_blotch(img, size, bx, by, radius_px, color, intensity):
    """Large soft-edged patch (Gaussian falloff) simulating a stain/bruise —
    brightness fades gradually over a wide area, unlike the sharp dots."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
    gauss = np.exp(-(dist ** 2) / (2 * radius_px ** 2)) * (intensity / 255.0)
    blend = gauss[..., None] * np.array(color, dtype=np.float32)
    img += blend


def add_fungal_dots(img, size, roi_center_px, roi_radius_px, severity, cfg, rng):
    dots_cfg = cfg["mock"]["fungal_dots"]
    max_field_radius_px = dots_cfg["placement_max_radius_frac"] * roi_radius_px

    if severity == "mild":
        m = dots_cfg["mild"]
        n_dots = int(rng.integers(m["n_dots_min"], m["n_dots_max"] + 1))
        cluster_specs = [(n_dots, m["cluster_spread_frac"] * size)]
    else:  # severe
        sv = dots_cfg["severe"]
        n_clusters = int(rng.integers(sv["n_clusters_min"], sv["n_clusters_max"] + 1))
        cluster_specs = [
            (
                int(rng.integers(sv["n_dots_per_cluster_min"], sv["n_dots_per_cluster_max"] + 1)),
                sv["cluster_spread_frac"] * size,
            )
            for _ in range(n_clusters)
        ]

    dot_r_min = dots_cfg["dot_radius_frac_min"] * size
    dot_r_max = dots_cfg["dot_radius_frac_max"] * size
    min_sep = dots_cfg["min_dot_separation_factor"] * (dot_r_min + dot_r_max)

    for n_dots, spread_px in cluster_specs:
        if rng.random() > dots_cfg["per_view_visibility_prob"]:
            continue  # this cluster isn't visible from this particular angle

        c_angle = rng.uniform(0, 2 * np.pi)
        c_r = rng.uniform(0, max_field_radius_px - spread_px) if max_field_radius_px > spread_px else 0
        cluster_center = (
            roi_center_px[0] + c_r * np.cos(c_angle),
            roi_center_px[1] + c_r * np.sin(c_angle),
        )
        positions = sample_cluster_positions(
            rng, cluster_center, spread_px, n_dots, min_sep, max_field_radius_px
        )
        for (bx, by) in positions:
            radius_px = rng.uniform(dot_r_min, dot_r_max)
            intensity = rng.uniform(dots_cfg["intensity_min"], dots_cfg["intensity_max"])
            draw_sharp_dot(
                img, size, bx, by, radius_px, dots_cfg["edge_transition_px"],
                dots_cfg["color"], intensity,
            )


def add_hard_negative_blotch(img, size, roi_center_px, roi_radius_px, cfg, rng):
    blotch_cfg = cfg["mock"]["hard_negative_blotch"]
    if rng.random() > blotch_cfg["per_view_visibility_prob"]:
        return  # stain happens to be out of frame / on the far side this angle

    angle = rng.uniform(0, 2 * np.pi)
    r_frac = rng.uniform(0, 0.5) * roi_radius_px
    bx = roi_center_px[0] + r_frac * np.cos(angle)
    by = roi_center_px[1] + r_frac * np.sin(angle)
    radius_px = rng.uniform(blotch_cfg["radius_frac_min"], blotch_cfg["radius_frac_max"]) * size
    intensity = rng.uniform(blotch_cfg["intensity_min"], blotch_cfg["intensity_max"])
    draw_blurry_blotch(img, size, bx, by, radius_px, blotch_cfg["color"], intensity)


def render_view(cfg, rng, sample_meta):
    img_cfg = cfg["image"]
    roi_cfg = cfg["roi"]
    mock = cfg["mock"]
    size = img_cfg["size_px"]

    mask, dist = circular_roi_mask_and_dist(size, roi_cfg["center_fraction"], roi_cfg["radius_fraction"])
    roi_radius_px = roi_cfg["radius_fraction"] * size
    roi_center_px = (roi_cfg["center_fraction"][0] * size, roi_cfg["center_fraction"][1] * size)

    bg = np.array(mock["background_level"], dtype=np.float32)
    base = np.array(mock["base_onion_color_mean"], dtype=np.float32)
    base = base + rng.normal(0, mock["base_onion_color_jitter"], size=3)
    brightness_scale = 1.0 + rng.uniform(
        -mock["per_view_brightness_jitter"], mock["per_view_brightness_jitter"]
    )
    base = base * brightness_scale

    img = np.tile(bg, (size, size, 1)).astype(np.float32)
    img[mask] = base

    noise = rng.normal(0, mock["base_noise_sigma"], size=(size, size, 3))
    img[mask] += noise[mask]

    # gentle center-bright / edge-dim falloff so the ROI doesn't look flat
    falloff = 1.0 - 0.25 * (dist / roi_radius_px)
    img = img * falloff[..., None]

    if sample_meta["compactdry"] == 1:
        add_fungal_dots(img, size, roi_center_px, roi_radius_px, sample_meta["severity"], cfg, rng)
    elif sample_meta["hard_negative"]:
        add_hard_negative_blotch(img, size, roi_center_px, roi_radius_px, cfg, rng)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["random_seed"])

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels_df = make_labels(cfg, rng)
    labels_df.to_csv(LABELS_PATH, index=False)
    print(f"Wrote {LABELS_PATH} ({len(labels_df)} rows, "
          f"{labels_df['compactdry'].sum()} positive / "
          f"{(labels_df['compactdry'] == 0).sum()} negative)")

    meta_df = assign_categories(labels_df, cfg, rng)
    meta_df.to_csv(METADATA_PATH, index=False)
    print(f"Wrote {METADATA_PATH} (debug-only breakdown):")
    print(f"  mild positive:   {(meta_df['severity'] == 'mild').sum()}")
    print(f"  severe positive: {(meta_df['severity'] == 'severe').sum()}")
    print(f"  hard-negative:   {meta_df['hard_negative'].sum()}")
    print(f"  clean negative:  {((meta_df['compactdry'] == 0) & ~meta_df['hard_negative']).sum()}")

    n_views = cfg["samples"]["n_views"]
    n_written = 0
    for _, row in meta_df.iterrows():
        for v in range(1, n_views + 1):
            img_arr = render_view(cfg, rng, row)
            out_path = IMAGES_DIR / f"{row['sample_code']}_V{v}.png"
            Image.fromarray(img_arr, mode="RGB").save(out_path)
            n_written += 1

    print(f"Wrote {n_written} images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
