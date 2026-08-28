"""
OnionGuard web app — Flask, single file.

Photos are taken on a PHONE (Pro/Manual mode, locked ISO/shutter/WB, in a
fixed jig) and uploaded from the browser. The server does not drive a
camera: prediction is Python (OpenCV + a pickled scikit-learn model) and
cannot run in the browser, so the page uploads the angles and this backend
classifies them.

    POST /capture           accept uploaded photo(s) for the session
    POST /predict-session   run the classifier over the uploaded angles
    GET  /capture/status    mode + how many angles are expected

Two details that decide whether the numbers mean anything:

  EXIF ORIENTATION  cv2.imread ignores the EXIF rotation flag, so a photo
                    the phone recorded as "portrait, rotate 90" would be
                    fed in sideways. Uploads are opened through Pillow with
                    exif_transpose applied before anything else.

  GEOMETRY          Every upload is centre-cropped square and resized to
                    config image.size_px. The blob detector's thresholds
                    are absolute pixel counts calibrated at that size, so a
                    raw 4000x3000 phone photo would make each of them mean
                    a different physical size.

Reading data (samples, stats, metrics) goes through web/data_source.py —
that's the seam for swapping in Supabase later.

Run:  ./venv/bin/python web/app.py     (listens on 0.0.0.0:5000)
"""
import base64
import io
import sys
import threading
import time
import uuid
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from PIL import Image, ImageOps, ExifTags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from advice import build_advice  # noqa: E402
from data_source import get_data_source, TABLE_FEATURE_COLUMNS  # noqa: E402
import garlic_anomaly as garlic_mod  # noqa: E402
from onion_detect import (detect_garlic_uv, detect_garlic_visible, detect_onion,  # noqa: E402
                           detect_onion_visible, normalize_to_onion)
import predict as predict_mod  # noqa: E402

try:
    import rawpy  # optional: only needed to accept RAW/DNG uploads
except ImportError:  # pragma: no cover - server may not have it installed
    rawpy = None

RAW_EXTENSIONS = {".dng", ".arw", ".nef", ".cr2", ".cr3", ".rw2", ".orf", ".raf"}

app = Flask(__name__)

REPORTS_DIR = ROOT / "reports"
CAPTURES_DIR = ROOT / "captures"
SESSION_TTL_SECONDS = 30 * 60

_EXIF_NAME = {v: k for k, v in ExifTags.TAGS.items()}

# Wording is fixed by the report and must not be paraphrased — in
# particular the positive label is "พบความผิดปกติที่สัมพันธ์กับเชื้อรา",
# never "พบเชื้อรา" (the system screens for an anomaly correlated with
# fungus, it does not identify fungus).
LABEL_TEXT = {
    1: "พบความผิดปกติที่สัมพันธ์กับเชื้อรา",
    0: "ไม่พบความผิดปกติ",
}
RESULT_DISCLAIMER = (
    "หมายเหตุ: เป็นระบบคัดกรองเบื้องต้น ไม่ใช่การยืนยันทางห้องปฏิบัติการ"
)

# Two clients share this backend and must not share storage.
#
#   research  the staff pages served from here. Rows carry a CompactDry YM
#             result and ARE the training set (src/export_features_from_db.py
#             reads them), so a stray row is a corrupted dataset.
#   public    the standalone AI site. No lab result exists for these scans and
#             never will, so they are kept apart and never trained on.
#
# This separates DATA, not PERMISSIONS: both write through the same
# service_role key, and nothing stops a determined caller from posting to the
# research route. Add auth to the staff pages if that ever matters.
DATASET_RESEARCH = "research"
DATASET_PUBLIC = "public"
DATASETS = {
    DATASET_RESEARCH: {
        "table": "scans",
        "bucket": "scans",
        "code_column": "sample_code",
        "requires_sample_code": True,
    },
    DATASET_PUBLIC: {
        "table": "public_scans",
        "bucket": "public-scans",
        "code_column": "scan_code",
        "requires_sample_code": False,
    },
}

# Which crops the staff site accepts, and HOW each one is screened.
#
# The two crops are screened by different kinds of model, because they have
# different kinds of data behind them:
#
#   onion   supervised classifier. Every training head carries a CompactDry
#           YM result, so the model learned positive from negative.
#   garlic  one-class anomaly detection. No garlic clove has a lab result —
#           every clove photographed so far is one that looked normal. What
#           is fitted is the spread of those normal cloves, and a new clove
#           is screened by how far outside it falls (src/garlic_anomaly.py).
#
# Running the shallot classifier on garlic instead was never an option: it
# has never seen a clove, so its label would be a number with no evidence
# behind it, printed next to the confidence figure the onion path earns
# honestly.
#
# Both report through the same LABEL_TEXT wording, which is what the anomaly
# detector can actually support: "พบความผิดปกติที่สัมพันธ์กับเชื้อรา" claims a
# deviation correlated with fungus, never an identification of fungus.
#
# public=True means the standalone AI site may offer the crop as well. It is
# not enough on its own: the public site only ever screens (its rows are never
# training data), so a crop appears there only while it HAS a working model —
# garlic drops off the public picker by itself whenever its baseline is
# unavailable, instead of quietly collecting photos nobody will train on.
CROP_ONION = "onion"
CROP_GARLIC = "garlic"
DEFAULT_CROP = CROP_ONION
CROPS = {
    CROP_ONION: {
        "label": "หอมแดง",
        "code_example": "S001",
        "subject_word": "หัวหอม",
        "has_model": True,
        "public": True,
    },
    CROP_GARLIC: {
        "label": "กระเทียม",
        "code_example": "GA001",
        "subject_word": "กลีบกระเทียม",
        # No supervised classifier exists (and cannot, from one class of
        # data); anomaly_model is what screens this crop instead.
        "has_model": False,
        "anomaly_model": True,
        "public": True,
        "anomaly_note": "กระเทียมตรวจโดยเทียบกับกลีบกระเทียมปกติที่เก็บไว้ — ระบบจะบอกว่า "
                        "“ไม่พบความผิดปกติ” หรือ “พบความผิดปกติที่สัมพันธ์กับเชื้อรา” "
                        "ยังไม่เคยเห็นกลีบที่ยืนยันด้วยผลแล็บว่าพบเชื้อ",
        # Shown, and used, only while no baseline is available yet.
        "collect_only_note": "ยังไม่มีค่าอ้างอิงกระเทียมปกติมากพอ ระบบจะเก็บภาพและค่าที่วัดได้ไว้เท่านั้น "
                             "ไม่มีการทำนายผล",
    },
}

_data_source = get_data_source()
_lock = threading.Lock()
_model = None
_cfg = None
_model_cfg = None
_sessions = {}


def get_model():
    global _model, _cfg, _model_cfg
    if _model is None:
        _cfg = predict_mod.load_config()
        _model_cfg = predict_mod.load_model_config()
        _model = predict_mod.joblib.load(predict_mod.MODEL_PATH)
    return _model, _cfg, _model_cfg


# The garlic baseline is not a pickled model, so it is not loaded with
# get_model(): it is either the frozen file that src/train_garlic_anomaly.py
# writes, or — when that file does not exist yet — fitted on demand from the
# normal cloves already stored in the database.
#
# Refitting is cached rather than done per scan, and re-checked on a timer so
# newly stored cloves eventually widen the baseline without a restart. A
# failure is cached too (for a shorter time), so a database outage does not
# turn every garlic scan into another timeout.
_GARLIC_TTL_OK = 15 * 60
_GARLIC_TTL_FAIL = 2 * 60
_garlic_cache = {"baseline": None, "reason": None, "checked": 0.0}
_garlic_lock = threading.Lock()


def _fit_garlic_baseline():
    """Fit the baseline from the garlic cloves already in the database.

    Only rows with no pred_label count. Those are the cloves photographed
    before this detector existed — the ones the operator described as normal
    garlic. A row this detector has already scored may be one of the
    anomalies it found, and feeding those back in would teach the baseline
    that anomalies are normal, widening it a little further with every flagged
    clove until nothing is unusual any more. A clove with a positive
    CompactDry result is excluded for the same reason.

    Returns (baseline, reason_it_is_missing).
    """
    client = getattr(_data_source, "client", None)
    if client is None:
        return None, "ยังไม่ได้ต่อฐานข้อมูล จึงยังไม่มีค่าอ้างอิงกระเทียมปกติ"
    try:
        rows = client.select(
            "scans",
            columns="sample_code,features,pred_label,compactdry_truth",
            filters={"crop": f"eq.{CROP_GARLIC}"},
        )
    except Exception as exc:  # noqa: BLE001 - screening degrades, never 500s
        return None, f"อ่านข้อมูลกระเทียมจากฐานข้อมูลไม่สำเร็จ: {str(exc)[:120]}"

    normals = [r["features"] for r in rows
               if r.get("features")
               and r.get("pred_label") is None
               and r.get("compactdry_truth") != 1]
    _, cfg, _ = get_model()
    try:
        baseline = garlic_mod.fit(normals, cfg=cfg,
                                  source=f"supabase:scans?crop={CROP_GARLIC} (auto)")
    except ValueError as exc:
        return None, str(exc)
    return baseline, None


def get_garlic_baseline():
    """(baseline, reason) — reason explains a None, for the operator to read."""
    _, cfg, _ = get_model()
    if not (cfg.get("garlic_anomaly") or {}).get("enabled", True):
        return None, "ปิดการตรวจความผิดปกติของกระเทียมไว้ใน config.json"

    now = time.time()
    with _garlic_lock:
        age = now - _garlic_cache["checked"]
        ttl = _GARLIC_TTL_OK if _garlic_cache["baseline"] else _GARLIC_TTL_FAIL
        if _garlic_cache["checked"] and age < ttl:
            return _garlic_cache["baseline"], _garlic_cache["reason"]

        # The frozen file wins whenever it exists: a deployed detector should
        # be the one that was reviewed and committed, not whatever the
        # database happened to hold this morning.
        baseline, reason = garlic_mod.load_baseline(), None
        if baseline is None:
            baseline, reason = _fit_garlic_baseline()
        _garlic_cache.update({"baseline": baseline, "reason": reason, "checked": now})
        return baseline, reason


# public_scans only gained its crop column in migration 002, and DDL cannot be
# run through the REST API — so this is a probe the app reads, not something it
# can fix. Until the column exists the public site keeps offering shallots
# only: a garlic row stored in a table that cannot say which crop it is would
# be indistinguishable from a shallot one, and the crop is the one thing that
# cannot be recomputed later.
_PUBLIC_CROP_TTL_OK = 15 * 60
_PUBLIC_CROP_TTL_FAIL = 2 * 60
_public_crop_cache = {"ok": None, "checked": 0.0}


def public_table_has_crop():
    client = getattr(_data_source, "client", None)
    if client is None:
        return False
    now = time.time()
    ttl = _PUBLIC_CROP_TTL_OK if _public_crop_cache["ok"] else _PUBLIC_CROP_TTL_FAIL
    if _public_crop_cache["ok"] is not None and now - _public_crop_cache["checked"] < ttl:
        return _public_crop_cache["ok"]
    try:
        client.select(DATASETS[DATASET_PUBLIC]["table"], columns="crop", limit=1)
        ok = True
    except Exception:  # noqa: BLE001 - a missing column reads as an error here
        ok = False
    _public_crop_cache.update({"ok": ok, "checked": now})
    return ok


def public_crops():
    """Every crop the public site lists, each flagged with whether it can be
    scanned RIGHT NOW.

    Unavailable crops are returned rather than hidden: a picker that silently
    loses an option looks broken, and the operator cannot tell a missing
    feature from a missing model. The page shows them disabled with the
    reason, and _capture refuses them, so the flag is what enforces this — not
    the list.

    Availability is one condition: the crop currently produces a verdict. The
    public site only ever screens, so a crop that could merely collect photos
    is not something to offer here — that is what public_scans exists to
    prevent. Which table column records the crop does NOT gate this; see
    _save_scan for where the crop goes when migration 002 has not been run.
    """
    out = []
    for c in crops_payload():
        if not c.get("public"):
            continue
        entry = dict(c)
        if c.get("screening") == "collect_only":
            entry["available"] = False
            entry["unavailable_reason"] = _collect_only_reason(c["id"])
        else:
            entry["available"] = True
        out.append(entry)
    return out


def available_public_crops():
    return {c["id"] for c in public_crops() if c.get("available")}


def _collect_only_reason(crop):
    """Why this scan produced no verdict, in the operator's words.

    For a crop screened by the anomaly detector, "no verdict" is always a
    missing baseline, and which reason it is (not enough normal cloves, no
    database, switched off) decides what the operator should do next — so the
    reason is shown, not just the fact.
    """
    note = CROPS[crop].get("collect_only_note", "")
    if CROPS[crop].get("anomaly_model"):
        _baseline, reason = get_garlic_baseline()
        if reason:
            return f"{note} — {reason}"
    return note


def crops_payload():
    """CROPS as the page needs it, with the note that actually applies now.

    Garlic's note depends on whether a baseline exists, which is a runtime
    fact — the template cannot work it out from a static dict.
    """
    out = []
    for cid, c in CROPS.items():
        entry = dict(id=cid, **c)
        if c.get("anomaly_model"):
            baseline, _reason = get_garlic_baseline()
            if baseline is not None:
                entry["note"] = "{} (ค่าอ้างอิงจากกลีบปกติ {} ตัวอย่าง)".format(
                    c.get("anomaly_note", ""), baseline.get("n_samples", 0))
                entry["screening"] = "anomaly"
            else:
                entry["note"] = _collect_only_reason(cid)
                entry["screening"] = "collect_only"
        else:
            entry["note"] = c.get("collect_only_note", "")
            entry["screening"] = "classifier" if c.get("has_model") else "collect_only"
        out.append(entry)
    return out


def _upload_cfg():
    _, cfg, _ = get_model()
    return cfg["capture_sequence"].get("upload", {})


with app.app_context():
    try:
        _u = get_model()[1]["capture_sequence"].get("upload", {})
        _max_mb = _u.get("max_file_mb", 25)
        _n = len(get_model()[1]["capture_sequence"]["steps"])
        app.config["MAX_CONTENT_LENGTH"] = int((_max_mb * _n + 10) * 1024 * 1024)
    except Exception:  # noqa: BLE001 - fall back to a sane ceiling
        app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024


def capture_steps():
    _, cfg, _ = get_model()
    return cfg["capture_sequence"]["steps"]


def _step_payload(index, steps, crop=None):
    """One capture step as the page needs it.

    The prompt names what is being photographed, and that word depends on the
    crop the operator picks AFTER this payload is sent — so both forms go out:
    `prompt` already filled in for one crop (the default, or `crop` when the
    caller knows it), and `prompt_template` with the {subject} placeholder
    intact so the page can refill it the moment the picker changes.
    """
    if index >= len(steps):
        return None
    step = steps[index]
    subject = CROPS[crop or DEFAULT_CROP]["subject_word"]
    return {
        "index": index,
        "id": step["id"],
        "kind": step.get("kind", "uv"),
        "prompt": step["prompt"].replace("{subject}", subject),
        "prompt_template": step["prompt"],
        "button": step.get("button", "เลือกภาพ"),
        "required": step.get("required", True),
        "step_number": index + 1,
        "total_steps": len(steps),
    }


def _prune_sessions():
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v["created"] > SESSION_TTL_SECONDS]:
        _sessions.pop(sid, None)


def _jpeg_data_uri(image_bgr, quality):
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _png_data_uri(image_bgr):
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _read_exif(pil_img):
    """Pull the shooting settings used to judge whether Pro mode was locked."""
    out = {}
    try:
        exif = pil_img.getexif()
        if not exif:
            return out
        ifd = exif.get_ifd(0x8769) or {}  # ExifIFD holds the exposure tags
        merged = {**dict(exif), **dict(ifd)}
        for name in ("ISOSpeedRatings", "ExposureTime", "FNumber", "WhiteBalance",
                     "ExposureProgram", "Model"):
            tag = _EXIF_NAME.get(name)
            if tag is not None and tag in merged:
                val = merged[tag]
                if isinstance(val, (Fraction,)):
                    val = float(val)
                elif isinstance(val, tuple) and len(val) == 2 and val[1]:
                    val = val[0] / val[1]
                out[name] = str(val)
    except Exception:  # noqa: BLE001 - EXIF is best-effort, never fatal
        pass
    return out


def _load_raw_to_bgr(data, cfg):
    """Decode a RAW/DNG upload to BGR, with every per-image auto-adjustment off.

    RAW is the better input here — the phone's JPEG pipeline applies noise
    reduction tuned to erase small bright specks, which is precisely the
    fungal-dot signal, plus sharpening and tone mapping that differ shot to
    shot. Decoding the RAW skips all of that.

    The postprocess settings are the whole point: no_auto_bright in
    particular MUST stay off, because rawpy would otherwise rescale each file
    independently to look pleasant, making two identical onions produce
    different brightness features.
    """
    if rawpy is None:
        raise ValueError(
            "เปิดไฟล์ RAW/DNG ไม่ได้ — เซิร์ฟเวอร์ยังไม่ได้ติดตั้ง rawpy "
            "(pip install rawpy) หรือให้ตั้งมือถือบันทึกเป็น JPEG แทน"
        )

    rc = cfg["capture_sequence"].get("upload", {}).get("raw", {})
    rgb = None
    source = "RAW/DNG"

    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            try:
                rgb = raw.postprocess(
                    no_auto_bright=rc.get("no_auto_bright", True),
                    use_camera_wb=rc.get("use_camera_wb", True),
                    use_auto_wb=False,
                    output_bps=rc.get("output_bps", 8),
                    gamma=(2.222, 4.5),  # sRGB-ish transfer, matching the JPEG path
                )
            except Exception:  # noqa: BLE001
                # Some DNGs (Lightroom "smart previews", other lossy/linear
                # variants) open but carry no Bayer data to demosaic. They
                # still hold an embedded JPEG, which is usable if big enough —
                # but it is a PROCESSED image, so it must be reported as such
                # rather than passed off as RAW.
                rgb = None
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"อ่านไฟล์ RAW/DNG ไม่สำเร็จ: {exc}") from None

    if rgb is None:
        # Re-open for the fallback: LibRaw's handle is left unusable by the
        # failed postprocess, so extracting the preview from it would report
        # "no preview" even when the file plainly has one.
        with rawpy.imread(io.BytesIO(data)) as raw2:
            rgb = _raw_embedded_preview(raw2, rc)
        source = "DNG (ภาพพรีวิวที่ฝังในไฟล์ ไม่ใช่ RAW แท้)"

    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

    # rawpy already applies the orientation flag, so no exif_transpose here.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    exif = {}
    try:
        pil = Image.open(io.BytesIO(data))  # DNG carries a readable TIFF/EXIF header
        exif = _read_exif(pil)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        pass
    exif["_source"] = source
    return bgr, exif


def _raw_embedded_preview(raw, rc):
    """Full-size JPEG embedded in a DNG that cannot be demosaiced.

    Rejected when too small: a thumbnail would silently hand the pipeline far
    less detail than an ordinary phone JPEG, and the resulting dot counts
    would look valid while being meaningless.
    """
    min_px = rc.get("min_preview_long_side_px", 1200)
    try:
        thumb = raw.extract_thumb()
    except Exception:  # noqa: BLE001
        raise ValueError(
            "ไฟล์ DNG นี้ไม่มีข้อมูล RAW ให้ประมวลผล และไม่มีภาพพรีวิวสำรอง "
            "— ให้ตั้งมือถือบันทึกเป็น DNG แบบเต็ม (RAW) หรือใช้ JPEG แทน"
        ) from None

    if thumb.format != rawpy.ThumbFormat.JPEG:
        raise ValueError("ไฟล์ DNG นี้ไม่มีข้อมูล RAW และพรีวิวไม่ใช่ JPEG จึงอ่านไม่ได้")

    arr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("ไฟล์ DNG นี้ไม่มีข้อมูล RAW และถอดรหัสพรีวิวไม่สำเร็จ")

    long_side = max(arr.shape[:2])
    if long_side < min_px:
        raise ValueError(
            f"ไฟล์ DNG นี้ไม่มีข้อมูล RAW มีแต่พรีวิวขนาดเล็ก ({long_side}px) "
            f"ซึ่งเล็กกว่าเกณฑ์ {min_px}px จึงใช้วัดผลไม่ได้ "
            "— น่าจะเป็น DNG แบบ smart preview ไม่ใช่ไฟล์จากกล้องโดยตรง"
        )
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _detect_subject(bgr, cfg, crop, kind):
    """Locate the subject with the detector its species needs.

    Both crops share the sanity bounds and the target framing scale: those
    describe the PHOTO and the geometry every downstream pixel threshold
    assumes, not the plant. Only the segmentation rule differs, because what
    makes a shallot stand out (red under UV, red skin in room light) is
    absent on a garlic clove — see onion_detect.detect_garlic_uv.
    """
    od = cfg["onion_detect"]
    sanity = od.get("sanity", {})
    lo = sanity.get("min_radius_frac_of_frame")
    hi = sanity.get("max_radius_frac_of_frame")

    if crop == CROP_GARLIC:
        gd = cfg.get("garlic_detect", {})
        if kind == "visible":
            return detect_garlic_visible(
                bgr, window_frac=gd.get("visible_window_frac", 0.55),
                min_area_frac=gd.get("visible_min_area_frac", 0.003),
                min_radius_frac=lo, max_radius_frac=hi)
        return detect_garlic_uv(
            bgr, k=gd.get("detect_k", 4.0), min_area_frac=gd.get("min_area_frac", 0.005),
            grow_factor=gd.get("uv_grow_factor", 0.6),
            min_radius_frac=lo, max_radius_frac=hi)

    if kind == "visible":
        return detect_onion_visible(
            bgr, k=od["visible_light_detect_k"], min_area_frac=od["min_area_frac"],
            min_radius_frac=lo, max_radius_frac=hi,
            min_abs_threshold=od.get("visible_light_min_abs_a", 6.0))
    return detect_onion(
        bgr, k=od["detect_k"], min_area_frac=od["min_area_frac"],
        min_radius_frac=lo, max_radius_frac=hi)


def _radius_frac_for(cfg, crop):
    """Fraction of the output frame the subject should occupy after re-framing."""
    if crop == CROP_GARLIC:
        return cfg.get("garlic_detect", {}).get("onion_radius_frac",
                                                cfg["onion_detect"]["onion_radius_frac"])
    return cfg["onion_detect"]["onion_radius_frac"]


def load_upload_to_bgr(file_storage, cfg, kind="uv", crop=DEFAULT_CROP):
    """Uploaded file -> (re-framed BGR array, exif dict, framing info, bg_level).

    Opened through Pillow so the EXIF orientation flag is applied; cv2 would
    silently ingest a portrait photo sideways. The onion is then located and
    the frame re-cropped so it is centred at a fixed scale — see
    src/onion_detect.py for why that matters more than it sounds.

    kind and crop together select the detector — see _detect_subject. A UV
    frame and a room-light frame need different logic, and so do a shallot
    and a garlic clove. No explicit alignment step is done between the
    two; both are independently centred/scaled around their own detected
    onion, which lines them up well enough in practice (verified against a
    real photo pair).

    bg_level is the median luminance OUTSIDE the onion, captured before
    re-framing. Comparing it across angles is how exposure drift is caught on
    phones that write no EXIF exposure data. Only meaningful for kind="uv"
    frames (a lit background has no such fixed baseline), but computed either
    way since it's cheap and callers filter by kind before comparing.
    """
    data = file_storage.read()
    ext = Path(file_storage.filename or "").suffix.lower()

    if ext in RAW_EXTENSIONS:
        bgr, exif = _load_raw_to_bgr(data, cfg)
    else:
        try:
            pil = Image.open(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            if ext in (".heic", ".heif"):
                raise ValueError(
                    "เปิดไฟล์ HEIC ไม่ได้ — ตั้งกล้องมือถือให้บันทึกเป็น JPEG "
                    "(iPhone: ตั้งค่า > กล้อง > รูปแบบ > เข้ากันได้มากที่สุด) "
                    "หรือติดตั้ง pillow-heif บนเซิร์ฟเวอร์"
                ) from None
            raise ValueError(f"เปิดไฟล์ภาพไม่ได้: {exc}") from None

        exif = _read_exif(pil)
        pil = ImageOps.exif_transpose(pil)  # honour the rotation flag
        bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    detection = _detect_subject(bgr, cfg, crop, kind)
    mask, _, _, info = detection

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_pixels = gray[~mask]
    bg_level = float(np.median(bg_pixels)) if bg_pixels.size else None

    frame, info = normalize_to_onion(bgr, cfg["image"]["size_px"],
                                      _radius_frac_for(cfg, crop), detection=detection)
    info["source_size"] = [bgr.shape[1], bgr.shape[0]]
    return frame, exif, info, bg_level


def check_out_of_distribution(features, model_cfg, tolerance=0.25):
    """Are the measured features inside the range the model was trained on?

    A tree ensemble cannot extrapolate. A value well beyond the largest one it
    saw during training falls in the same leaf as that largest value, so the
    predicted probability simply stops responding — it does not keep rising,
    and it gives no signal that the input was unusual. Real photos have
    already produced 71 dots where the training maximum was 18, so this is not
    hypothetical.

    tolerance is how far outside the observed range still counts as fine,
    expressed as a fraction of that range.
    """
    ranges = model_cfg.get("training_feature_ranges")
    if not ranges:
        return {"status": "unknown",
                "message": "โมเดลนี้ไม่ได้บันทึกช่วงข้อมูลตอนฝึกไว้ จึงตรวจไม่ได้"}

    outliers = []
    for name, r in ranges.items():
        v = features.get(name)
        if v is None:
            continue
        span = r["max"] - r["min"]
        # Skip features that were effectively constant in training. Their span
        # can be a floating-point crumb rather than a true zero, which would
        # make any different value look infinitely far outside the range and
        # produce a spurious warning (A_low did exactly this: 0.13-0.13).
        scale = max(abs(r["max"]), abs(r["min"]), 1e-9)
        if span <= 1e-6 or span / scale < 1e-3:
            continue
        margin = span * tolerance
        if v > r["max"] + margin:
            outliers.append((name, v, r["min"], r["max"], (v - r["max"]) / span))
        elif v < r["min"] - margin:
            outliers.append((name, v, r["min"], r["max"], (r["min"] - v) / span))

    if not outliers:
        return {"status": "in_range",
                "message": "ค่าที่วัดได้อยู่ในช่วงที่โมเดลเคยเห็นตอนฝึก"}

    outliers.sort(key=lambda o: -o[4])

    def _fmt(x, lo, hi):
        # Enough significant digits that a genuinely tight training range does
        # not print as a single repeated value (A_low spans 0.1252-0.1286 and
        # read as "0.13-0.13" at two decimals, which looked like a bug in the
        # check rather than a real distribution gap).
        span = abs(hi - lo)
        digits = 2 if span >= 0.05 else (3 if span >= 0.005 else 4)
        return f"{x:.{digits}f}"

    detail = "; ".join(
        f"{n} = {_fmt(v, lo, hi)} (ตอนฝึก {_fmt(lo, lo, hi)}–{_fmt(hi, lo, hi)})"
        for n, v, lo, hi, _ in outliers[:3]
    )
    return {
        "status": "out_of_range",
        "n_outliers": len(outliers),
        "detail": detail,
        "message": f"ค่าที่วัดได้ {len(outliers)} ตัวอยู่นอกช่วงที่โมเดลเคยเห็นตอนฝึก ({detail}) "
                   "— โมเดลประมาณค่านอกช่วงที่เคยเห็นไม่ได้ ผลและเปอร์เซ็นต์ที่ได้จึงเชื่อถือไม่ได้",
    }


def check_scale_consistency(captures, cfg):
    """Do the angles agree on how big the onion is?

    All four photos are the same onion, so the detected radius should barely
    change between them. A large spread means either the phone-to-onion
    distance moved a lot between shots (hand-held drift, which the
    re-framing corrects but which also changes perspective) or one angle's
    detection failed. Either way the operator should know before the result
    is trusted.
    """
    sanity = cfg["onion_detect"].get("sanity", {})
    limit = sanity.get("max_radius_spread_across_views")
    if not limit:
        return {"status": "disabled"}

    # With a single-view protocol there is nothing to cross-check. Say so
    # explicitly — reporting "unknown" every scan would read as a fault.
    if len(captures) < 2:
        return {"status": "not_applicable",
                "message": "ถ่ายมุมเดียว จึงไม่มีมุมอื่นให้เทียบขนาด"}

    vals = [(c["step_id"], (c.get("framing") or {}).get("radius_frac_of_frame"))
            for c in captures]
    known = [(s, v) for s, v in vals if v]
    if len(known) < 2:
        return {"status": "unknown", "message": "ข้อมูลไม่พอสำหรับเทียบขนาดระหว่างมุม"}

    r = [v for _, v in known]
    lo, hi = min(r), max(r)
    spread = (hi - lo) / hi if hi else 0.0
    detail = ", ".join(f"{s}={v:.2f}" for s, v in known)

    if spread > limit:
        return {
            "status": "inconsistent", "spread": round(spread, 3), "radii": detail,
            "message": f"ขนาดหัวหอมที่ตรวจได้ต่างกัน {spread*100:.0f}% ระหว่างมุม ({detail}) "
                       f"— เกินเกณฑ์ {limit*100:.0f}% อาจถือมือถือห่างไม่เท่ากัน "
                       "หรือมีมุมที่ตรวจจับพลาด",
        }
    return {
        "status": "consistent", "spread": round(spread, 3), "radii": detail,
        "message": f"ขนาดหัวหอมใกล้เคียงกันทุกมุม (ต่างกัน {spread*100:.0f}%)",
    }


def check_background_consistency(captures, cfg):
    """Did the phone hold its exposure across the angles?

    Used instead of EXIF because the tested Android phone reports no exposure
    tags at all. The background is a fixed surface, so a spread in its
    brightness between angles means the camera re-exposed and the brightness
    features are no longer comparable across angles.
    """
    bc = cfg["onion_detect"].get("background_consistency", {})
    if not bc.get("enabled", True):
        return {"status": "disabled"}

    # Single-view protocol: exposure cannot drift "between angles" when there
    # is only one. Stated plainly instead of as an unknown.
    if len(captures) < 2:
        return {"status": "not_applicable",
                "message": "ถ่ายมุมเดียว จึงไม่มีมุมอื่นให้เทียบ exposure "
                           "(ยังควรล็อก Pro mode ให้เหมือนกันทุกหัวที่เก็บ)"}

    levels = [(c["step_id"], c.get("bg_level")) for c in captures]
    known = [(s, v) for s, v in levels if v is not None]
    if len(known) < 2:
        return {"status": "unknown", "message": "ข้อมูลพื้นหลังไม่พอสำหรับเปรียบเทียบ"}

    vals = [v for _, v in known]
    lo, hi = min(vals), max(vals)
    detail = ", ".join(f"{s}={v:.1f}" for s, v in known)

    # Exposure scales the image MULTIPLICATIVELY, so compare a ratio, not a
    # difference. The background here is near-black (median ~9), where even a
    # 40% exposure change moves the level by only ~4 counts — an absolute
    # threshold silently passes real drift.
    min_level = bc.get("min_level_to_judge", 4.0)
    if lo < min_level:
        return {
            "status": "unknown", "levels": detail,
            "message": f"พื้นหลังมืดเกินไป (ต่ำสุด {lo:.1f}) จนเทียบ exposure ไม่ได้ ({detail})",
        }

    ratio = hi / lo
    limit = bc.get("max_ratio", 1.25)
    if ratio > limit:
        return {
            "status": "inconsistent", "ratio": round(ratio, 3), "levels": detail,
            "message": f"ความสว่างพื้นหลังต่างกัน {ratio:.2f} เท่าระหว่างมุม ({detail}) "
                       f"— เกินเกณฑ์ {limit} เท่า แปลว่ามือถือปรับ exposure เอง "
                       "ค่าความสว่างเทียบข้ามมุมไม่ได้",
        }
    return {
        "status": "consistent", "ratio": round(ratio, 3), "levels": detail,
        "message": f"ความสว่างพื้นหลังใกล้เคียงกันทุกมุม (ต่างกัน {ratio:.2f} เท่า) "
                   "— exposure น่าจะถูกล็อกไว้",
    }


def check_exif_consistency(captures, cfg):
    """Compare shooting settings across the uploaded angles.

    Differing ISO/shutter means the phone re-exposed between shots, i.e. Pro
    mode was not locked — the brightness features stop being comparable.
    Returns a dict the UI can show; never blocks.
    """
    ec = cfg["capture_sequence"].get("upload", {}).get("exif_check", {})
    if not ec.get("enabled", True):
        return {"status": "disabled"}

    tags = ec.get("compare_tags", ["ISOSpeedRatings", "ExposureTime"])
    per_tag = {}
    missing = 0
    for c in captures:
        e = c.get("exif") or {}
        if not e:
            missing += 1
            continue
        for t in tags:
            if t in e:
                per_tag.setdefault(t, set()).add(e[t])

    if missing == len(captures) or not per_tag:
        return {
            "status": "unknown",
            "message": "อ่าน EXIF ไม่ได้ (ไฟล์อาจถูกลบ metadata หรือเป็น PNG) "
                       "จึงตรวจไม่ได้ว่าล็อกค่ากล้องไว้จริงหรือไม่",
        }

    varying = {t: sorted(v) for t, v in per_tag.items() if len(v) > 1}
    if varying:
        detail = "; ".join(f"{t}: {', '.join(v)}" for t, v in varying.items())
        return {
            "status": "inconsistent",
            "varying": varying,
            "message": f"ค่ากล้องไม่เท่ากันระหว่างมุม ({detail}) — "
                       "แปลว่าไม่ได้ล็อก Pro mode ค่าความสว่างเทียบข้ามมุมไม่ได้",
        }

    return {
        "status": "consistent",
        "settings": {t: sorted(v)[0] for t, v in per_tag.items()},
        "message": "ค่ากล้องเท่ากันทุกมุม (ล็อก Pro mode ไว้เรียบร้อย)",
    }


@app.after_request
def add_cors_headers(response):
    """Allow the Vercel-hosted static frontend to call this API cross-origin.

    No cookies/credentials are used anywhere in this app, so reflecting any
    origin (or "*") carries no session-hijack risk — it only lets a browser
    read responses that were already public over plain HTTP.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def no_cache_dynamic(response):
    """Stop the browser caching pages, the stylesheet, and API responses.

    The page's JavaScript is inline in the template, so a cached copy keeps
    running OLD logic against the CURRENT API — which already caused a real
    confusion: the server was returning an out-of-range warning while the
    browser showed a page built before that warning existed, so the result
    looked clean when it was not. Safety warnings must never be served from a
    stale cache. Data URIs in responses are unaffected.

    JSON is on the list for the same reason, learned the same way: every API
    response here describes what the server can do RIGHT NOW — which crops
    have a working model, what the capture protocol is, what a scan measured.
    None of it carries a validator, so a browser is free to reuse an old copy
    for as long as it likes. A phone did exactly that and kept showing a
    one-crop picker for an hour after the second crop went live, with no way
    for the person holding it to tell.
    """
    if response.mimetype in ("text/html", "text/css", "application/javascript",
                             "application/json"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.context_processor
def inject_globals():
    """Every template gets these — the mock banner must appear on all pages."""
    return {
        "is_mock": _data_source.is_mock_data(),
        "project_title": "ระบบ AI คัดกรองเชื้อราบนหอมแดงด้วยแสง UV 365 นาโนเมตร",
        "disclaimer": RESULT_DISCLAIMER,
    }


@app.route("/")
@app.route("/scan")
def scan():
    steps = capture_steps()
    return render_template("scan.html", active="scan",
                            steps=[_step_payload(i, steps) for i in range(len(steps))],
                            total_steps=len(steps),
                            crops=crops_payload(),
                            default_crop=DEFAULT_CROP)


@app.route("/samples")
def samples():
    return render_template(
        "samples.html", active="samples",
        samples=_data_source.get_samples(),
        feature_columns=TABLE_FEATURE_COLUMNS,
        label_text=LABEL_TEXT,
        crop_labels={cid: c["label"] for cid, c in CROPS.items()},
    )


@app.route("/dataset")
def dataset():
    return render_template("dataset.html", active="dataset",
                            stats=_data_source.get_dataset_stats())


@app.route("/model")
def model():
    return render_template("model.html", active="model",
                            metrics=_data_source.get_model_metrics(),
                            info=_data_source.get_model_info())


@app.route("/label")
def label():
    """Enter the CompactDry culture result for scans that have none yet.

    Kept separate from /scan because the two happen days apart: the photo is
    taken up front, the culture is read after incubation. Scans still waiting
    for a result are listed first, since those are the ones blocking the
    dataset from being usable for training.
    """
    return render_template("label.html", active="label")


@app.route("/api/site-info")
def api_site_info():
    """Chrome data every static page needs — mock banner, title, disclaimer."""
    return jsonify({
        "is_mock": _data_source.is_mock_data(),
        "project_title": "ระบบ AI คัดกรองเชื้อราบนหอมแดงด้วยแสง UV 365 นาโนเมตร",
        "disclaimer": RESULT_DISCLAIMER,
    })


@app.route("/api/samples")
def api_samples():
    return jsonify({
        "samples": _data_source.get_samples(),
        "feature_columns": TABLE_FEATURE_COLUMNS,
        "label_text": LABEL_TEXT,
    })


@app.route("/api/dataset-stats")
def api_dataset_stats():
    crop = request.args.get("crop", DEFAULT_CROP)
    if crop not in CROPS:
        return jsonify({"error": f"ชนิดพืชไม่ถูกต้อง: {crop}"}), 400
    return jsonify({"stats": _data_source.get_dataset_stats(crop), "crop": crop})


@app.route("/api/model-metrics")
def api_model_metrics():
    return jsonify({
        "metrics": _data_source.get_model_metrics(),
        "info": _data_source.get_model_info(),
    })


@app.route("/api/scans")
def api_scans():
    client = getattr(_data_source, "client", None)
    if client is None:
        return jsonify({"error": "ยังไม่ได้ต่อ Supabase", "rows": []}), 503
    # ?crop=garlic narrows the list to one species. Without it every crop is
    # listed, because the lab result is entered from this same page and a
    # staff member reading cultures has both species in front of them.
    crop = request.args.get("crop")
    if crop is not None and crop not in CROPS:
        return jsonify({"error": f"ชนิดพืชไม่ถูกต้อง: {crop}", "rows": []}), 400
    try:
        rows = client.select(
            "scans",
            columns="id,sample_code,crop,captured_at,pred_label,pred_conf,compactdry_truth,"
                    "truth_recorded_at,ood_status,borderline,image_path",
            order="captured_at.desc",
            filters={"crop": f"eq.{crop}"} if crop else None,
        )
        return jsonify({"rows": rows, "crops": {cid: c["label"] for cid, c in CROPS.items()}})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)[:200], "rows": []}), 500


@app.route("/api/public-scans")
def api_public_scans():
    """History for the standalone AI site.

    Reads public_scans only. The staff endpoints (/api/samples, /api/scans)
    expose the research set, including CompactDry results — that belongs to
    the staff pages, not to a public scanner.
    """
    client = getattr(_data_source, "client", None)
    if client is None:
        return jsonify({"error": "ยังไม่ได้ต่อ Supabase", "rows": []}), 503
    try:
        limit = min(int(request.args.get("limit", 200)), 500)
    except ValueError:
        limit = 200
    try:
        columns = ("id,scan_code,captured_at,pred_label,pred_conf,pred_proba,"
                   "ood_status,borderline,framing_ok")
        # Same two homes as _save_scan writes to, read back under one name so
        # the page does not have to know which era a row was stored in.
        columns += ",crop" if public_table_has_crop() else ",crop:quality_notes->>crop"
        rows = client.select(
            DATASETS[DATASET_PUBLIC]["table"],
            columns=columns,
            order="captured_at.desc",
            limit=limit,
        )
        return jsonify({"rows": rows, "label_text": LABEL_TEXT,
                        "crops": {c["id"]: c["label"] for c in public_crops()},
                        "default_crop": DEFAULT_CROP,
                        "disclaimer": RESULT_DISCLAIMER})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)[:200], "rows": []}), 500


@app.route("/api/scan-image")
def api_scan_image():
    """Mint a short-lived signed URL for a scan's stored photo and redirect
    to it. The bucket is private (see supabase_client.signed_url), so the
    label page cannot just link to the path directly — it has to go through
    the server, which holds the key that can sign it."""
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "ไม่ได้ระบุ path ของรูป"}), 400

    client = getattr(_data_source, "client", None)
    if client is None:
        return jsonify({"error": "ยังไม่ได้ต่อ Supabase"}), 503

    try:
        url = client.signed_url("scans", path)
        if not url:
            return jsonify({"error": "สร้างลิงก์รูปไม่สำเร็จ"}), 500
        return redirect(url)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)[:200]}), 500


@app.route("/api/label", methods=["POST"])
def api_label():
    """Record (or clear) the lab result for one scan."""
    client = getattr(_data_source, "client", None)
    if client is None:
        return jsonify({"error": "ยังไม่ได้ต่อ Supabase"}), 503

    body = request.get_json(silent=True) or {}
    scan_id = body.get("id")
    truth = body.get("compactdry_truth")
    if scan_id is None:
        return jsonify({"error": "ไม่ได้ระบุ id"}), 400
    if truth not in (0, 1, None):
        return jsonify({"error": "compactdry_truth ต้องเป็น 0, 1 หรือ null"}), 400

    try:
        patch = {"compactdry_truth": truth,
                 "truth_recorded_at": "now()" if truth is not None else None}
        client.update("scans", {"id": scan_id}, patch)
        return jsonify({"ok": True, "id": scan_id, "compactdry_truth": truth})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)[:200]}), 500


@app.route("/health")
def health():
    """Cheap liveness probe. Deliberately does NOT touch the model or the
    database — it exists to keep a sleeping free-tier host awake, so it must
    stay fast and must not fail when a dependency is briefly down."""
    return jsonify({"status": "ok", "service": "onionguard"})


def _start_keepalive():
    """Keep the free-tier host awake so a demo never hits a ~50s cold start.

    Render's free plan spins a service down after ~15 min with no INBOUND
    traffic, and only inbound HTTP counts — so the app pings its own public
    URL (Render injects RENDER_EXTERNAL_URL) every few minutes from a daemon
    thread. Absent that env var (local runs, other hosts) this is a no-op,
    so nothing changes off Render.

    Failures are swallowed: this is best-effort warmth, never a hard
    dependency, and it must not crash the worker if the network blips.
    """
    import os
    import urllib.request

    base = os.environ.get("RENDER_EXTERNAL_URL")
    if not base:
        return

    url = base.rstrip("/") + "/health"
    interval = int(os.environ.get("KEEPALIVE_INTERVAL_SECONDS", "600"))

    def _loop():
        while True:
            time.sleep(interval)
            try:
                urllib.request.urlopen(url, timeout=30).read()
            except Exception:  # noqa: BLE001 - best effort, never fatal
                pass

    threading.Thread(target=_loop, name="keepalive", daemon=True).start()


_start_keepalive()


@app.route("/reports/<path:filename>")
def reports(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/capture/status")
def capture_status():
    """Capture protocol + the crop list for the caller's site.

    ?dataset=public narrows the crops to the ones the standalone AI site may
    offer (see public_crops). The staff page asks without it and gets every
    crop, including any that can only collect data at the moment.
    """
    _, cfg, _ = get_model()
    steps = capture_steps()
    u = _upload_cfg()
    public = request.args.get("dataset") == DATASET_PUBLIC
    return jsonify({
        "mode": "phone_upload",
        "total_steps": len(steps),
        "steps": [_step_payload(i, steps) for i in range(len(steps))],
        "allowed_extensions": u.get("allowed_extensions", []),
        "max_file_mb": u.get("max_file_mb", 25),
        "target_size": cfg["image"]["size_px"],
        "default_crop": DEFAULT_CROP,
        "crops": public_crops() if public else crops_payload(),
    })


@app.route("/capture", methods=["POST"])
def capture():
    return _capture(DATASET_RESEARCH)


@app.route("/public/capture", methods=["POST"])
def public_capture():
    return _capture(DATASET_PUBLIC)


def _capture(dataset):
    """Accept uploaded photo(s) for the session.

    Files are assigned to consecutive steps from the session's current
    position, so the page can send all angles at once or one at a time.
    retake=true drops the previous upload and re-fills that slot.

    dataset decides which table the finished scan lands in, and is fixed by
    the route rather than read from the request — see DATASETS.
    """
    _prune_sessions()
    session_id = request.form.get("session_id") or None
    retake = request.form.get("retake") == "true"
    sample_code = (request.form.get("sample_code") or "").strip()

    crop = (request.form.get("crop") or DEFAULT_CROP).strip()
    if crop not in CROPS:
        return jsonify({"error": f"ชนิดพืชไม่ถูกต้อง: {crop}"}), 400
    # The public site's crop list is a runtime fact (see public_crops), so the
    # page can be holding a stale one. Refusing is deliberate: silently
    # scanning the clove as a shallot instead would hand back a confident
    # verdict from a model that has never seen garlic.
    if dataset == DATASET_PUBLIC and crop not in available_public_crops():
        return jsonify({"error": f"ตอนนี้เว็บสาธารณะยังคัดกรอง{CROPS[crop]['label']}ไม่ได้ "
                                 "กรุณาโหลดหน้านี้ใหม่แล้วลองอีกครั้ง"}), 400

    if DATASETS[dataset]["requires_sample_code"]:
        # Required for research scans: one whose sample code is unknown can
        # never be matched to its CompactDry culture result, which makes it
        # useless as training data no matter how good the image is.
        if not sample_code:
            return jsonify({"error": "กรุณากรอกรหัสตัวอย่าง (เช่น "
                                     f"{CROPS[crop]['code_example']}) ก่อนอัปโหลด"}), 400
    elif not sample_code:
        # Public scans get a generated code instead. Asking a member of the
        # public to invent one produced exactly the junk it sounds like
        # ("hhh", "test"), and the code has no meaning without a lab result
        # to match it to anyway.
        sample_code = "P" + datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:6]
    if len(sample_code) > 32:
        return jsonify({"error": "รหัสตัวอย่างยาวเกินไป (สูงสุด 32 ตัวอักษร)"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "ไม่พบไฟล์ภาพที่อัปโหลด"}), 400

    steps = capture_steps()
    _, cfg, _ = get_model()
    target = cfg["image"]["size_px"]
    u = _upload_cfg()
    allowed = {e.lower() for e in u.get("allowed_extensions", [])}

    if not session_id or session_id not in _sessions:
        session_id = uuid.uuid4().hex[:12]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _sessions[session_id] = {
            "created": time.time(),
            "dir": CAPTURES_DIR / f"{stamp}_{session_id}",
            "captured": [],
            "step_index": 0,
            "sample_code": sample_code,
            "dataset": dataset,
            "crop": crop,
        }
        retake = False

    session = _sessions[session_id]
    # The dataset is set when the session is created and never changes after:
    # continuing a research session through the public route (or the reverse)
    # would silently move a scan into the wrong table.
    if session.get("dataset") != dataset:
        return jsonify({"error": "เซสชันนี้เริ่มจากอีกระบบหนึ่ง กรุณาเริ่มใหม่"}), 400
    # Same reasoning for the crop: switching it half way through would file
    # one head's photos under a species it is not, and that row is training
    # data for whichever model reads it later.
    if session.get("crop", DEFAULT_CROP) != crop:
        return jsonify({"error": "เซสชันนี้เริ่มจากพืชอีกชนิดหนึ่ง กรุณาเริ่มใหม่"}), 400
    session["sample_code"] = sample_code

    if retake:
        if not session["captured"]:
            return jsonify({"error": "ยังไม่มีภาพให้เลือกใหม่"}), 400
        session["captured"].pop()
        session["step_index"] = max(0, session["step_index"] - 1)

    remaining = len(steps) - session["step_index"]
    if remaining <= 0:
        return jsonify({"error": "ครบทุกมุมแล้ว"}), 400
    if len(files) > remaining:
        return jsonify({"error": f"เลือกมาเกินจำนวนที่เหลือ (เหลือ {remaining} มุม แต่ส่งมา {len(files)} ไฟล์)"}), 400

    accepted = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if allowed and ext not in allowed:
            return jsonify({"error": f"ไฟล์ไม่รองรับ: {f.filename} ({ext})"}), 400

        step = steps[session["step_index"]]
        try:
            frame, exif, info, bg_level = load_upload_to_bgr(
                f, cfg, kind=step.get("kind", "uv"), crop=crop)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        saved_path = None
        if cfg["capture_sequence"].get("save_captures", True):
            session["dir"].mkdir(parents=True, exist_ok=True)
            saved_path = session["dir"] / f"{step['id']}.png"
            cv2.imwrite(str(saved_path), frame)

        session["captured"].append({
            "step_id": step["id"],
            "kind": step.get("kind", "uv"),
            "path": str(saved_path) if saved_path else None,
            "exif": exif,
            "bg_level": bg_level,
            "framing": info,
        })
        session["step_index"] += 1

        # Framing problems are reported per photo so a bad shot can be
        # replaced now rather than quietly degrading the result.
        warnings = []
        subject = CROPS[crop].get("subject_word", "หัวหอม")
        if not info.get("ok"):
            warnings.append(f"หาตำแหน่ง{subject}ไม่สำเร็จ ({info.get('reason', 'ไม่ทราบสาเหตุ')}) "
                            "— ใช้กรอบกลางภาพแทน ผลอาจคลาดเคลื่อน")
        if info.get("touches_edge"):
            warnings.append(f"{subject}ชนขอบภาพด้าน " + ", ".join(info["touches_edge"]) +
                            " อาจถ่ายไม่ครบทั้งหัว")
        pad_warn = cfg["onion_detect"].get("warn_pad_area_frac", 0.25)
        if info.get("padded"):
            black = float((frame.max(axis=2) == 0).mean())
            if black > pad_warn:
                warnings.append(f"{subject}อยู่ชิดขอบภาพ ต้องเติมพื้นที่ดำ {black*100:.0f}% "
                                f"— จัดให้{subject}อยู่กลางกรอบมากขึ้น")

        accepted.append({
            "id": step["id"],
            "kind": step.get("kind", "uv"),
            "preview": _jpeg_data_uri(frame, cfg["capture_sequence"]["preview"].get("jpeg_quality", 92)),
            "exif": exif,
            "source_name": f.filename,
            "framing_ok": bool(info.get("ok")),
            "warnings": warnings,
        })

    next_step = _step_payload(session["step_index"], steps, session.get("crop"))
    return jsonify({
        "session_id": session_id,
        "accepted": accepted,
        "was_retake": retake,
        "next_step": next_step,
        "all_captured": next_step is None,
    })


@app.route("/capture/skip", methods=["POST"])
@app.route("/public/capture/skip", methods=["POST"])
def capture_skip():
    """Skip the current step — only allowed when it is marked required:false.

    Today that is the visible-light cross-check: the model must still
    produce a screening result from the UV photo alone (visible_features.py
    already returns a sentinel for "no visible photo"), so an operator who
    cannot re-light the box should not be blocked from finishing the scan.
    """
    _prune_sessions()
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "ไม่พบเซสชัน กรุณาเริ่มใหม่"}), 400

    steps = capture_steps()
    idx = session["step_index"]
    if idx >= len(steps):
        return jsonify({"error": "ครบทุกขั้นตอนแล้ว"}), 400

    step = steps[idx]
    if step.get("required", True):
        return jsonify({"error": f"ขั้นตอนนี้ ({step['id']}) จำเป็นต้องมีภาพ ข้ามไม่ได้"}), 400

    session["step_index"] += 1
    next_step = _step_payload(session["step_index"], steps, session.get("crop"))
    return jsonify({
        "session_id": session_id,
        "skipped": step["id"],
        "next_step": next_step,
        "all_captured": next_step is None,
    })


def _save_scan(session, result, checks):
    """Persist the scan to Supabase, image included.

    Which table and bucket depends on session["dataset"] — see DATASETS for
    why the two are kept apart.

    Never raises: a database or network problem must not lose the operator's
    result, which is already computed and on screen. The response carries
    saved_to_db so a silent failure is visible rather than assumed.
    """
    client = getattr(_data_source, "client", None)
    if client is None:
        return {"ok": False, "reason": "ไม่ได้ต่อ Supabase (ใช้ไฟล์ในเครื่อง)"}

    target = DATASETS[session.get("dataset", DATASET_RESEARCH)]
    table, bucket, code_col = target["table"], target["bucket"], target["code_column"]

    try:
        uv = [c for c in session["captured"] if c["kind"] == "uv"]
        image_path = None
        if uv and uv[0].get("path"):
            src = Path(uv[0]["path"])
            if src.exists():
                image_path = f"{session['dir'].name}/{src.name}"
                client.upload(bucket, image_path, src.read_bytes(), content_type="image/png")

        payload = {
            code_col: session.get("sample_code") or session["dir"].name,
            "image_path": image_path,
            "features": result["features"],
            "framing_ok": not checks["framing_failed"],
            "quality_notes": {
                "scale": checks["scale"], "background": checks["background"],
                "exif": checks["exif"],
                "framing_failed_views": checks["framing_failed"],
            },
        }
        # Which crop was scanned is the one thing in a row that cannot be
        # recomputed later, so it is always recorded — the only question is
        # where. public_scans gets a real column once migration 002 has been
        # run (queryable, constrained); until then it goes in quality_notes,
        # which is jsonb and already there. Sending a column that does not
        # exist fails the whole insert and loses the scan, so the two cases
        # are kept apart rather than hoping.
        crop = session.get("crop", DEFAULT_CROP)
        if table == "scans" or public_table_has_crop():
            payload["crop"] = crop
        else:
            payload["quality_notes"]["crop"] = crop
        # A crop with no model produces no prediction, so every model-derived
        # column stays null. Writing 0 or "unknown" instead would make an
        # unscored row indistinguishable from one the model called negative.
        if checks.get("label") is not None:
            payload.update({
                "pred_label": int(checks["label"]),
                "pred_conf": round(float(result["confidence"]), 4),
                "pred_proba": round(float(result["proba_positive"]), 4),
                "decision_threshold": round(float(result["decision_threshold"]), 4),
                "borderline": bool(result["borderline"]),
            })
            # Out-of-distribution is a classifier-only check: it compares the
            # features against the ranges the shallot model was fitted on.
            if checks.get("ood") is not None:
                payload["ood_status"] = checks["ood"].get("status")
                payload["quality_notes"]["ood"] = checks["ood"]
            # An anomaly row stores its real score and threshold here. The
            # pred_proba column above is the same decision expressed 0-1
            # (0.5 == exactly on the threshold) so the two crops' rows can be
            # listed together, but the raw score is what a later reviewer
            # needs to see, and it does not fit in a 0-1 column.
            if checks.get("anomaly") is not None:
                payload["quality_notes"]["anomaly"] = checks["anomaly"]
        if table == "scans":
            payload["image_original"] = session["dir"].name
            # left null on purpose: filled in after the CompactDry culture is
            # read. null means "not tested yet", never "no fungus".
            payload["compactdry_truth"] = None

        row = client.insert(table, payload)
        return {"ok": True, "id": row.get("id"), "sample_code": row.get(code_col)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:200]}


def _result_images(measured, cfg):
    """(overlay data-URI, clean data-URI, source Path) for one measured head.

    Shared by every scan path so the operator sees the same two images
    whichever model produced the verdict.
    """
    overlay_uri = _png_data_uri(measured["overlay_image"])
    clean_uri = None
    src = Path(measured["overlay_source_path"])
    if src.exists():
        clean_img = cv2.imread(str(src))
        if clean_img is not None:
            clean_uri = _jpeg_data_uri(
                clean_img, cfg["capture_sequence"]["preview"].get("jpeg_quality", 92))
    return overlay_uri, clean_uri, src


def _anomaly_session(session, session_id, crop, paths, visible_path, cfg, uv, baseline):
    """Finish a scan for a crop screened by the one-class detector.

    Measures exactly as every other path does, then asks how far the clove
    sits outside the normal cloves' spread. The out-of-distribution check is
    NOT run: it compares against the shallot model's feature ranges and would
    report "out of range" for every garlic clove while saying nothing about
    this one. The baseline comparison here is that check, done against the
    right reference set.
    """
    try:
        with _lock:
            measured = predict_mod.measure_head(paths, cfg=cfg, visible_image_path=visible_path)
        scored = garlic_mod.score(measured["features"], baseline)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"ประมวลผลไม่สำเร็จ: {exc}"}), 500

    overlay_uri, clean_uri, src = _result_images(measured, cfg)
    if overlay_uri is None:
        return jsonify({"error": "สร้างภาพ overlay ไม่สำเร็จ"}), 500

    exif_report = check_exif_consistency(uv, cfg)
    bg_report = check_background_consistency(uv, cfg)
    scale_report = check_scale_consistency(uv, cfg)
    framing_failed = [c["step_id"] for c in uv
                      if not (c.get("framing") or {}).get("ok", True)]

    top = [{"feature": f["feature"], "label": f["label"], "z": round(f["z"], 2),
            "value": f["value"], "median": f["median"], "direction": f["direction"]}
           for f in scored["top_features"]]
    anomaly_note = {
        "score": round(scored["score"], 3),
        "threshold": round(scored["threshold"], 3),
        "n_baseline_samples": scored["n_baseline_samples"],
        "baseline_created_at": scored["baseline_created_at"],
        "baseline_source": scored["baseline_source"],
        "top_features": top,
        "missing_features": scored["missing"],
    }

    # Shaped like a classifier result so _save_scan writes both crops the
    # same way — see the comment there on what pred_proba means for a row
    # scored this way.
    stored = {
        "features": measured["features"],
        "confidence": scored["confidence"],
        "proba_positive": scored["proba_anomaly"],
        "decision_threshold": 0.5,
        "borderline": scored["borderline"],
    }
    saved = _save_scan(session, stored, {
        "label": scored["label"], "ood": None, "anomaly": anomaly_note,
        "scale": scale_report, "background": bg_report, "exif": exif_report,
        "framing_failed": framing_failed,
    })
    _sessions.pop(session_id, None)

    expected = cfg["samples"]["n_views"]
    return jsonify({
        "screening": "anomaly",
        "crop": crop,
        "crop_label": CROPS[crop]["label"],
        "saved_to_db": saved,
        "sample_code": session.get("sample_code"),
        "label": scored["label"],
        "label_text": LABEL_TEXT[scored["label"]],
        "confidence": scored["confidence"],
        "confidence_pct": round(scored["confidence"] * 100, 1),
        "borderline": scored["borderline"],
        "anomaly": anomaly_note,
        "processing_time": round(measured["processing_time"], 2),
        "overlay_image": overlay_uri,
        "clean_image": clean_uri,
        "overlay_source": src.name,
        "n_small_blobs": measured["features"].get("n_small_sharp_blobs_viewmax", 0),
        "n_large_blotches": measured["features"].get("n_large_blotches_viewmax", 0),
        "n_views_used": len(uv),
        "expected_views": expected,
        "partial_views": len(uv) < expected,
        "exif_check": exif_report,
        "background_check": bg_report,
        "scale_check": scale_report,
        "framing_failed_views": framing_failed,
        "disclaimer": RESULT_DISCLAIMER,
        "is_mock_data": _data_source.is_mock_data(),
    })


def _collect_only_session(session, session_id, crop, paths, visible_path, cfg, uv):
    """Finish a scan for a crop that has no model: measure and store, no label.

    The image quality checks that still mean something are kept — framing,
    scale and background all describe the PHOTO and hold whatever the
    species. The out-of-distribution check is dropped on purpose: it compares
    against feature ranges recorded from the shallot training set, so running
    it on garlic would report "out of range" for every clove and say nothing
    about the photo.
    """
    try:
        with _lock:
            measured = predict_mod.measure_head(paths, cfg=cfg, visible_image_path=visible_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"วัดค่าจากภาพไม่สำเร็จ: {exc}"}), 500

    overlay_uri, clean_uri, src = _result_images(measured, cfg)

    exif_report = check_exif_consistency(uv, cfg)
    bg_report = check_background_consistency(uv, cfg)
    scale_report = check_scale_consistency(uv, cfg)
    framing_failed = [c["step_id"] for c in uv
                      if not (c.get("framing") or {}).get("ok", True)]

    saved = _save_scan(session, measured, {
        "label": None, "ood": None, "anomaly": None, "scale": scale_report,
        "background": bg_report, "exif": exif_report,
        "framing_failed": framing_failed,
    })
    _sessions.pop(session_id, None)

    return jsonify({
        "collect_only": True,
        "crop": crop,
        "crop_label": CROPS[crop]["label"],
        "collect_only_note": _collect_only_reason(crop),
        "saved_to_db": saved,
        "sample_code": session.get("sample_code"),
        "processing_time": round(measured["processing_time"], 2),
        "overlay_image": overlay_uri,
        "clean_image": clean_uri,
        "overlay_source": src.name,
        "n_small_blobs": measured["features"].get("n_small_sharp_blobs_viewmax", 0),
        "n_large_blotches": measured["features"].get("n_large_blotches_viewmax", 0),
        "n_features": len(measured["features"]),
        "n_views_used": len(uv),
        "exif_check": exif_report,
        "background_check": bg_report,
        "scale_check": scale_report,
        "framing_failed_views": framing_failed,
        "disclaimer": RESULT_DISCLAIMER,
        "is_mock_data": _data_source.is_mock_data(),
    })


@app.route("/predict-session", methods=["POST"])
def predict_session():
    return _predict_session(DATASET_RESEARCH)


@app.route("/public/predict-session", methods=["POST"])
def public_predict_session():
    return _predict_session(DATASET_PUBLIC)


def _predict_session(dataset):
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "ไม่พบเซสชัน กรุณาเริ่มใหม่"}), 400
    if session.get("dataset", DATASET_RESEARCH) != dataset:
        return jsonify({"error": "เซสชันนี้เริ่มจากอีกระบบหนึ่ง กรุณาเริ่มใหม่"}), 400

    # Only UV frames go to the classifier. Dark frames (if the protocol adds
    # them) are kept but NOT used — dark-frame subtraction is not implemented.
    # The visible-light frame (if any) is NOT a UV angle either — it never
    # feeds predict_head as a primary image, only as the optional cross-check.
    uv = [c for c in session["captured"] if c["kind"] == "uv"]
    visible = [c for c in session["captured"] if c["kind"] == "visible"]
    if not uv:
        return jsonify({"error": "ยังไม่มีภาพ UV ในเซสชันนี้"}), 400

    model, cfg, model_cfg = get_model()
    paths = [c["path"] for c in uv if c["path"]]
    if len(paths) != len(uv):
        return jsonify({"error": "การบันทึกภาพไม่ครบ กรุณาเปิด save_captures ใน config"}), 500
    # A visible-light frame whose onion was never found got a fallback centred
    # crop, so it is at the wrong scale and does not line up with the UV frame.
    # Dropping it here makes predict fall back to the "no visible photo"
    # sentinel — which is what extract_features.py already does on the training
    # side, and the only way the two stay comparable.
    visible_usable = (visible and visible[0].get("path")
                      and (visible[0].get("framing") or {}).get("ok", True))
    visible_path = visible[0]["path"] if visible_usable else None

    crop = session.get("crop", DEFAULT_CROP)
    if CROPS[crop].get("anomaly_model"):
        # No baseline yet (too few normal cloves stored, or no database) —
        # the scan still measures and stores, exactly as before this detector
        # existed. Screening against a baseline fitted on a handful of cloves
        # would be worse than not screening.
        baseline, _reason = get_garlic_baseline()
        if baseline is not None:
            return _anomaly_session(session, session_id, crop, paths, visible_path,
                                    cfg, uv, baseline)
        return _collect_only_session(session, session_id, crop, paths, visible_path, cfg, uv)
    if not CROPS[crop]["has_model"]:
        return _collect_only_session(session, session_id, crop, paths, visible_path, cfg, uv)

    try:
        with _lock:
            result = predict_mod.predict_head(paths, model=model, cfg=cfg, model_cfg=model_cfg,
                                              visible_image_path=visible_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"ประมวลผลไม่สำเร็จ: {exc}"}), 500

    # The clean frame is the same one WITHOUT the circles, so the operator can
    # toggle the markings off and judge the fluorescence itself rather than
    # only seeing what the detector decided to ring.
    overlay_uri, clean_uri, src = _result_images(result, cfg)
    if overlay_uri is None:
        return jsonify({"error": "สร้างภาพ overlay ไม่สำเร็จ"}), 500

    label = result["label"]
    expected = cfg["samples"]["n_views"]
    exif_report = check_exif_consistency(uv, cfg)
    bg_report = check_background_consistency(uv, cfg)
    scale_report = check_scale_consistency(uv, cfg)
    ood_report = check_out_of_distribution(result["features"], model_cfg)
    framing_failed = [c["step_id"] for c in uv
                      if not (c.get("framing") or {}).get("ok", True)]

    advice_cfg = cfg.get("advice", {})
    advice = build_advice(
        label,
        borderline=result["borderline"],
        ood_status=ood_report.get("status"),
        framing_ok=not framing_failed,
        scale_status=scale_report.get("status"),
        background_status=bg_report.get("status"),
        advice_table=advice_cfg.get("messages"),
    ) if advice_cfg.get("enabled", True) else None

    saved = _save_scan(session, result, {
        "label": label, "ood": ood_report, "anomaly": None, "scale": scale_report,
        "background": bg_report, "exif": exif_report,
        "framing_failed": framing_failed,
    })

    _sessions.pop(session_id, None)

    return jsonify({
        "screening": "classifier",
        "crop": crop,
        "crop_label": CROPS[crop]["label"],
        "saved_to_db": saved,
        "sample_code": session.get("sample_code"),
        "label": label,
        "label_text": LABEL_TEXT[label],
        "confidence": result["confidence"],
        "confidence_pct": round(result["confidence"] * 100, 1),
        "processing_time": round(result["processing_time"], 2),
        "overlay_image": overlay_uri,
        "clean_image": clean_uri,
        "overlay_source": src.name,
        "n_small_blobs": result["features"].get("n_small_sharp_blobs_viewmax", 0),
        "n_large_blotches": result["features"].get("n_large_blotches_viewmax", 0),
        "n_views_used": len(uv),
        "expected_views": expected,
        "partial_views": len(uv) < expected,
        "exif_check": exif_report,
        "background_check": bg_report,
        "scale_check": scale_report,
        "ood_check": ood_report,
        "proba_positive": round(result["proba_positive"], 3),
        "decision_threshold": round(result["decision_threshold"], 3),
        "borderline": result["borderline"],
        "framing_failed_views": framing_failed,
        "advice": advice,
        "disclaimer": RESULT_DISCLAIMER,
        "is_mock_data": _data_source.is_mock_data(),
    })


if __name__ == "__main__":
    import os

    # debug is OFF by default and must be opted into explicitly. This app binds
    # 0.0.0.0 so the whole LAN can reach it, and Werkzeug's debug mode ships an
    # interactive console that permits arbitrary code execution on any traceback.
    #   ONIONGUARD_DEBUG=1 python web/app.py
    debug = os.environ.get("ONIONGUARD_DEBUG") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug, threaded=True)
