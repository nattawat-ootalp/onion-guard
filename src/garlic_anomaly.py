"""
Garlic screening as ONE-CLASS anomaly detection.

Why this is not the onion path
------------------------------
The shallot head is a supervised classifier: every training row carries a
CompactDry YM result, so the model learned the difference between a positive
and a negative head. No garlic row has a lab result — every garlic clove
photographed so far is simply a garlic clove that looked normal. That is one
class, not two, and a two-class model cannot be fitted from it.

What is fitted instead is the SHAPE OF NORMAL: for each feature, where the
normal cloves sit (median) and how much they naturally vary (MAD). A new
clove is scored by how far outside that spread it falls. Nothing here has
ever seen a fungal clove, so the result is deliberately worded as the
project's standing phrase — "พบความผิดปกติที่สัมพันธ์กับเชื้อรา", an anomaly
correlated with fungus — and never as an identification of fungus.

Robust statistics, not mean/SD
------------------------------
Median and MAD are used because the training set is small (tens of cloves)
and unaudited: a single mis-framed photo shifts a mean and inflates an SD
enough to swallow a real anomaly, while the median barely moves.

Scoring
-------
    z_i    = (x_i - median_i) / scale_i        (clipped, see DIRECTIONS)
    score  = max_i z_i
    label  = 1 when score >= threshold

max, not a sum: one feature far outside normal IS the finding, and summing
would let twelve slightly-off features outvote it. It also makes the result
explainable — the feature that produced the max is the reason, and it is
reported back to the operator.

The threshold is calibrated on the training cloves themselves (a high
percentile of their own scores, times a margin), so it means "further out
than the normal cloves ever were", not an arbitrary constant.

No scikit-learn, no pickle: the fitted object is a small JSON file of
medians and scales. It loads on the Pi, it diffs in git, and a human can
read why a clove was flagged.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "garlic_anomaly.json"

# Which measured features describe "a normal clove", and which way a
# deviation is allowed to count.
#
#   "high"  only an INCREASE is suspicious. Fewer fluorescent dots than the
#           normal cloves is a cleaner clove, not a stranger one — counting
#           it as an anomaly would flag the best samples in the batch.
#   "both"  either direction is odd. These describe how the clove looks
#           overall (colour, brightness spread, texture); a clove far from
#           normal on any of them is worth a second look whichever way it went.
#
# Head-level names (_viewmean) because that is what predict.measure_head
# returns. The protocol currently shoots one angle per clove, so _viewmean
# and _viewmax are the same number; _viewmean is used so the set keeps
# meaning what it says if the protocol goes back to several angles.
#
# Deliberately excluded: mean_R/G/B and brightness_mean (they track exposure
# more than the clove), ratio_small_to_large (degenerate — nearly 1.0 on
# every clove), and uv_exclusive_dot_frac (carries a sentinel value when no
# ordinary-light photo was taken, which would score as a huge deviation for
# a reason that has nothing to do with the clove).
DIRECTIONS = {
    "n_small_sharp_blobs_viewmean": "high",
    "cluster_density_viewmean": "high",
    "n_large_blotches_viewmean": "high",
    "blob_max_viewmean": "high",
    "avg_small_blob_diam_px_viewmean": "high",
    "A_high_viewmean": "high",
    "A_low_viewmean": "high",
    "texture_viewmean": "both",
    "brightness_sd_viewmean": "both",
    "F_p95_viewmean": "both",
    "NDFI_mean_viewmean": "both",
    "NDFI_p95_viewmean": "both",
}

# Thai names shown to the operator when a feature is the reason for a flag.
FEATURE_LABELS = {
    "n_small_sharp_blobs_viewmean": "จำนวนจุดเรืองแสงเล็กขอบคม",
    "cluster_density_viewmean": "ความหนาแน่นของกลุ่มจุด",
    "n_large_blotches_viewmean": "จำนวนปื้นขนาดใหญ่",
    "blob_max_viewmean": "ขนาดจุดที่ใหญ่ที่สุด",
    "avg_small_blob_diam_px_viewmean": "ขนาดเฉลี่ยของจุดเล็ก",
    "A_high_viewmean": "สัดส่วนพื้นที่สว่างผิดปกติ",
    "A_low_viewmean": "สัดส่วนพื้นที่มืดผิดปกติ",
    "texture_viewmean": "ความหยาบของพื้นผิว",
    "brightness_sd_viewmean": "ความไม่สม่ำเสมอของความสว่าง",
    "F_p95_viewmean": "ความสว่างช่วงบน (p95)",
    "NDFI_mean_viewmean": "ดัชนีสีเรืองแสงเฉลี่ย (NDFI)",
    "NDFI_p95_viewmean": "ดัชนีสีเรืองแสงช่วงบน (NDFI p95)",
}

# Every number the detector needs, in one place. config.json > garlic_anomaly
# overrides any of them; the defaults are here so the module works standalone.
DEFAULTS = {
    # Below this many normal cloves the spread of "normal" is not measurable
    # and the detector refuses to run at all (the scan falls back to
    # collect-only) rather than screening against noise.
    "min_samples": 12,
    # A feature is kept only if this fraction of the training cloves actually
    # carry it — a feature present in three rows out of sixty is not a baseline.
    "min_feature_coverage": 0.8,
    # Threshold = percentile of the training cloves' own scores x margin,
    # floored at min_threshold. The percentile (not the max) keeps one odd
    # training clove from raising the bar for everything after it; the margin
    # keeps a clove that merely ties the worst normal one from being flagged.
    "threshold_percentile": 97.5,
    "threshold_margin": 1.10,
    "min_threshold": 3.5,
    # A single feature cannot contribute more than this, so one absurd value
    # (a failed measurement, a 0-scale feature) cannot pin the score at
    # infinity and drown out what the other features say.
    "max_z": 12.0,
    # Ceiling for a feature that never varied at all across the normal cloves
    # (n_large_blotches is 0 on every one of them). Its "spread" is a floor,
    # not a measurement, so the z it produces is arbitrary in size — it is
    # kept above the threshold, because a value normal cloves never showed IS
    # evidence, but capped below max_z so it cannot outrank a feature whose
    # spread was actually measured.
    "constant_feature_z": 6.0,
    # Floors for the per-feature scale, applied in this order of preference:
    # MAD -> spread of the middle 80% -> a fraction of the median -> absolute.
    # Without them a feature that is constant across all training cloves
    # (n_large_blotches = 0 everywhere) would divide by zero and flag every
    # later clove that has even one blotch.
    "scale_floor_frac": 0.05,
    "scale_floor_abs": 1e-9,
    # Turns the score into a 0-1 figure for display and storage. 0.5 sits
    # exactly on the threshold by construction; k sets how fast confidence
    # grows as the score moves away from it.
    "confidence_k": 2.5,
    # A result this close to 0.5 (i.e. this close to the threshold) is
    # reported as borderline: the clove sits where the baseline cannot
    # separate "unusual" from "the far edge of normal".
    "borderline_band": 0.1,
    # Reported back as "why", when a feature is at least this far out.
    "explain_min_z": 1.5,
    "explain_max_features": 3,
}


# ---------------------------------------------------------------- helpers

def params(cfg=None):
    """DEFAULTS overridden by config.json > garlic_anomaly (if present)."""
    out = dict(DEFAULTS)
    block = (cfg or {}).get("garlic_anomaly") or {}
    for k, v in block.items():
        if k in out:
            out[k] = v
    return out


def feature_directions(cfg=None):
    block = (cfg or {}).get("garlic_anomaly") or {}
    return dict(block.get("features") or DIRECTIONS)


def _percentile(sorted_vals, pct):
    """Linear-interpolated percentile. numpy would do this in one call, but
    this module is imported by the Pi-side code path, which keeps its
    dependency list to what predict.py already needs."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def lookup(features, name):
    """Read one feature, accepting the plain per-view name as a fallback.

    A live scan's feature dict carries both "texture" and "texture_viewmean";
    a features.csv exported for training carries only the plain name. Both
    are the same measurement while the protocol shoots one angle per clove,
    so a baseline fitted from either file scores the other correctly instead
    of reporting every feature as missing.
    """
    if name in features:
        return features[name]
    for suffix in ("_viewmean", "_viewmax"):
        if name.endswith(suffix):
            return features.get(name[: -len(suffix)])
    return None


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------- fitting

def fit(feature_rows, cfg=None, source="unknown"):
    """Fit the shape of normal from cloves that are ALL assumed normal.

    feature_rows: list of feature dicts, exactly as predict.measure_head
    returns them and as the scans table stores them.

    Returns the baseline dict (JSON-serialisable), or raises ValueError when
    there are too few usable cloves to describe normal with.
    """
    p = params(cfg)
    directions = feature_directions(cfg)

    rows = [r for r in feature_rows if r]
    if len(rows) < p["min_samples"]:
        raise ValueError(
            f"มีกระเทียมปกติเพียง {len(rows)} ตัวอย่าง ต้องมีอย่างน้อย {p['min_samples']} "
            "ตัวอย่างจึงจะวัด “ช่วงปกติ” ได้")

    features = {}
    for name, direction in directions.items():
        vals = sorted(v for v in (_finite(lookup(r, name)) for r in rows) if v is not None)
        if len(vals) < p["min_feature_coverage"] * len(rows):
            continue

        median = _percentile(vals, 50)
        mad = _percentile(sorted(abs(v - median) for v in vals), 50)
        # 1.4826 makes MAD comparable to an SD on normally distributed data,
        # so a z of 3 here means roughly what a z of 3 means anywhere else.
        scale = 1.4826 * mad
        if scale <= 0:
            # p10-p90 spread, converted to the same SD-equivalent units.
            scale = (_percentile(vals, 90) - _percentile(vals, 10)) / 2.563
        constant = scale <= 0
        if scale <= 0:
            scale = p["scale_floor_frac"] * abs(median)
        if scale <= 0:
            scale = p["scale_floor_abs"]

        features[name] = {
            "direction": direction,
            # True when every training clove measured the same value, so the
            # scale below is a floor rather than an observed spread — score()
            # caps what such a feature may contribute.
            "constant": constant,
            "median": median,
            "scale": scale,
            "mad": mad,
            "p10": _percentile(vals, 10),
            "p90": _percentile(vals, 90),
            "n": len(vals),
        }

    if not features:
        raise ValueError("ไม่มีฟีเจอร์ใดที่ครบพอจะใช้เป็นค่าอ้างอิงได้")

    baseline = {
        "version": 1,
        "crop": "garlic",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "n_samples": len(rows),
        "params": p,
        "features": features,
        # Filled in below — score() needs a baseline dict to work on, so the
        # threshold is computed from a copy that carries a provisional one.
        "threshold": p["min_threshold"],
    }

    scores = sorted(score(r, baseline)["score"] for r in rows)
    threshold = max(_percentile(scores, p["threshold_percentile"]) * p["threshold_margin"],
                    p["min_threshold"])
    baseline["threshold"] = threshold
    baseline["train_scores"] = {
        "p50": _percentile(scores, 50),
        "p90": _percentile(scores, 90),
        "p97_5": _percentile(scores, 97.5),
        "max": scores[-1],
    }
    # How many of the training cloves the finished detector would flag. These
    # are known-normal, so this is the in-sample false alarm count and belongs
    # in the file next to the threshold it justifies.
    baseline["train_flagged"] = sum(1 for s in scores if s >= threshold)
    return baseline


# ---------------------------------------------------------------- scoring

def score(features, baseline):
    """Score one measured clove against the baseline.

    Returns {score, threshold, label, confidence, proba_anomaly,
             per_feature, top_features, missing}. label is 1 when an anomaly
    was found, matching web/app.py LABEL_TEXT.
    """
    p = params({"garlic_anomaly": baseline.get("params", {})})
    threshold = float(baseline.get("threshold", p["min_threshold"]))

    per_feature, missing = [], []
    for name, spec in baseline["features"].items():
        x = _finite(lookup(features, name))
        if x is None:
            missing.append(name)
            continue
        z = (x - spec["median"]) / (spec["scale"] or p["scale_floor_abs"])
        z = max(z, 0.0) if spec["direction"] == "high" else abs(z)
        z = min(z, p["constant_feature_z"] if spec.get("constant") else p["max_z"])
        per_feature.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": x,
            "median": spec["median"],
            "z": z,
            "direction": spec["direction"],
        })

    if not per_feature:
        raise ValueError("ค่าที่วัดได้ไม่มีฟีเจอร์ที่ตรงกับค่าอ้างอิงเลย")

    per_feature.sort(key=lambda d: d["z"], reverse=True)
    s = per_feature[0]["z"]
    label = int(s >= threshold)

    # Logistic on the log-ratio of score to threshold: 0.5 exactly at the
    # threshold, rising towards 1 above it and falling towards 0 below. A
    # linear map would put "score 0" and "score just under threshold" at
    # visibly different confidences for no evidential reason.
    ratio = math.log((s + 1e-6) / threshold)
    proba = 1.0 / (1.0 + math.exp(-p["confidence_k"] * ratio))
    confidence = proba if label == 1 else 1.0 - proba

    top = [f for f in per_feature[:p["explain_max_features"]]
           if f["z"] >= p["explain_min_z"]]

    return {
        "score": s,
        "threshold": threshold,
        "label": label,
        "confidence": confidence,
        "proba_anomaly": proba,
        "borderline": abs(proba - 0.5) <= p["borderline_band"],
        "per_feature": per_feature,
        "top_features": top,
        "missing": missing,
        "n_baseline_samples": baseline.get("n_samples"),
        "baseline_created_at": baseline.get("created_at"),
        "baseline_source": baseline.get("source"),
    }


# ---------------------------------------------------------------- storage

def load_baseline(path=BASELINE_PATH):
    """Read a frozen baseline, or None when none has been written."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        baseline = json.load(f)
    if not baseline.get("features"):
        return None
    return baseline


def save_baseline(baseline, path=BASELINE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
