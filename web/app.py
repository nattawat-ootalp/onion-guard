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
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps, ExifTags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_source import get_data_source, TABLE_FEATURE_COLUMNS  # noqa: E402
from onion_detect import detect_onion, normalize_to_onion  # noqa: E402
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


def _step_payload(index, steps):
    if index >= len(steps):
        return None
    step = steps[index]
    return {
        "index": index,
        "id": step["id"],
        "kind": step.get("kind", "uv"),
        "prompt": step["prompt"],
        "button": step.get("button", "เลือกภาพ"),
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


def load_upload_to_bgr(file_storage, cfg):
    """Uploaded file -> (re-framed BGR array, exif dict, framing info, bg_level).

    Opened through Pillow so the EXIF orientation flag is applied; cv2 would
    silently ingest a portrait photo sideways. The onion is then located and
    the frame re-cropped so it is centred at a fixed scale — see
    src/onion_detect.py for why that matters more than it sounds.

    bg_level is the median luminance OUTSIDE the onion, captured before
    re-framing. Comparing it across angles is how exposure drift is caught on
    phones that write no EXIF exposure data.
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

    od = cfg["onion_detect"]
    sanity = od.get("sanity", {})
    detection = detect_onion(
        bgr, k=od["detect_k"], min_area_frac=od["min_area_frac"],
        min_radius_frac=sanity.get("min_radius_frac_of_frame"),
        max_radius_frac=sanity.get("max_radius_frac_of_frame"),
    )
    mask, _, _, info = detection

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_pixels = gray[~mask]
    bg_level = float(np.median(bg_pixels)) if bg_pixels.size else None

    frame, info = normalize_to_onion(bgr, cfg["image"]["size_px"],
                                      od["onion_radius_frac"], detection=detection)
    info["source_size"] = [bgr.shape[1], bgr.shape[0]]
    return frame, exif, info, bg_level


def check_out_of_distribution(features, model_cfg, tolerance=0.25):
    """Are the measured features inside the range the model was trained on?

    A Random Forest cannot extrapolate. A value well beyond the largest one it
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
                   "— Random Forest ประมาณค่านอกช่วงไม่ได้ ผลและเปอร์เซ็นต์ที่ได้จึงเชื่อถือไม่ได้",
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
def no_cache_html(response):
    """Stop the browser caching pages and the stylesheet.

    The page's JavaScript is inline in the template, so a cached copy keeps
    running OLD logic against the CURRENT API — which already caused a real
    confusion: the server was returning an out-of-range warning while the
    browser showed a page built before that warning existed, so the result
    looked clean when it was not. Safety warnings must never be served from a
    stale cache. Data URIs in responses are unaffected.
    """
    if response.mimetype in ("text/html", "text/css", "application/javascript"):
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
                            total_steps=len(steps))


@app.route("/samples")
def samples():
    return render_template(
        "samples.html", active="samples",
        samples=_data_source.get_samples(),
        feature_columns=TABLE_FEATURE_COLUMNS,
        label_text=LABEL_TEXT,
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


@app.route("/reports/<path:filename>")
def reports(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/capture/status")
def capture_status():
    _, cfg, _ = get_model()
    steps = capture_steps()
    u = _upload_cfg()
    return jsonify({
        "mode": "phone_upload",
        "total_steps": len(steps),
        "steps": [_step_payload(i, steps) for i in range(len(steps))],
        "allowed_extensions": u.get("allowed_extensions", []),
        "max_file_mb": u.get("max_file_mb", 25),
        "target_size": cfg["image"]["size_px"],
    })


@app.route("/capture", methods=["POST"])
def capture():
    """Accept uploaded photo(s) for the session.

    Files are assigned to consecutive steps from the session's current
    position, so the page can send all angles at once or one at a time.
    retake=true drops the previous upload and re-fills that slot.
    """
    _prune_sessions()
    session_id = request.form.get("session_id") or None
    retake = request.form.get("retake") == "true"

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
        }
        retake = False

    session = _sessions[session_id]

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
            frame, exif, info, bg_level = load_upload_to_bgr(f, cfg)
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
        if not info.get("ok"):
            warnings.append(f"หาตำแหน่งหัวหอมไม่สำเร็จ ({info.get('reason', 'ไม่ทราบสาเหตุ')}) "
                            "— ใช้กรอบกลางภาพแทน ผลอาจคลาดเคลื่อน")
        if info.get("touches_edge"):
            warnings.append("หัวหอมชนขอบภาพด้าน " + ", ".join(info["touches_edge"]) +
                            " อาจถ่ายไม่ครบทั้งหัว")
        pad_warn = cfg["onion_detect"].get("warn_pad_area_frac", 0.25)
        if info.get("padded"):
            black = float((frame.max(axis=2) == 0).mean())
            if black > pad_warn:
                warnings.append(f"หัวหอมอยู่ชิดขอบภาพ ต้องเติมพื้นที่ดำ {black*100:.0f}% "
                                "— จัดให้หัวหอมอยู่กลางกรอบมากขึ้น")

        accepted.append({
            "id": step["id"],
            "kind": step.get("kind", "uv"),
            "preview": _jpeg_data_uri(frame, cfg["capture_sequence"]["preview"].get("jpeg_quality", 92)),
            "exif": exif,
            "source_name": f.filename,
            "framing_ok": bool(info.get("ok")),
            "warnings": warnings,
        })

    next_step = _step_payload(session["step_index"], steps)
    return jsonify({
        "session_id": session_id,
        "accepted": accepted,
        "was_retake": retake,
        "next_step": next_step,
        "all_captured": next_step is None,
    })


@app.route("/predict-session", methods=["POST"])
def predict_session():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "ไม่พบเซสชัน กรุณาเริ่มใหม่"}), 400

    # Only UV frames go to the classifier. Dark frames (if the protocol adds
    # them) are kept but NOT used — dark-frame subtraction is not implemented.
    uv = [c for c in session["captured"] if c["kind"] == "uv"]
    if not uv:
        return jsonify({"error": "ยังไม่มีภาพ UV ในเซสชันนี้"}), 400

    model, cfg, model_cfg = get_model()
    paths = [c["path"] for c in uv if c["path"]]
    if len(paths) != len(uv):
        return jsonify({"error": "การบันทึกภาพไม่ครบ กรุณาเปิด save_captures ใน config"}), 500

    try:
        with _lock:
            result = predict_mod.predict_head(paths, model=model, cfg=cfg, model_cfg=model_cfg)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"ประมวลผลไม่สำเร็จ: {exc}"}), 500

    overlay_uri = _png_data_uri(result["overlay_image"])
    if overlay_uri is None:
        return jsonify({"error": "สร้างภาพ overlay ไม่สำเร็จ"}), 500

    # The same frame WITHOUT the circles, so the operator can toggle the
    # markings off and judge the fluorescence itself rather than only seeing
    # what the detector decided to ring.
    clean_uri = None
    src = Path(result["overlay_source_path"])
    if src.exists():
        clean_img = cv2.imread(str(src))
        if clean_img is not None:
            clean_uri = _jpeg_data_uri(
                clean_img, cfg["capture_sequence"]["preview"].get("jpeg_quality", 92))

    label = result["label"]
    expected = cfg["samples"]["n_views"]
    exif_report = check_exif_consistency(uv, cfg)
    bg_report = check_background_consistency(uv, cfg)
    scale_report = check_scale_consistency(uv, cfg)
    ood_report = check_out_of_distribution(result["features"], model_cfg)
    framing_failed = [c["step_id"] for c in uv
                      if not (c.get("framing") or {}).get("ok", True)]

    _sessions.pop(session_id, None)

    return jsonify({
        "label": label,
        "label_text": LABEL_TEXT[label],
        "confidence": result["confidence"],
        "confidence_pct": round(result["confidence"] * 100, 1),
        "processing_time": round(result["processing_time"], 2),
        "overlay_image": overlay_uri,
        "clean_image": clean_uri,
        "overlay_source": Path(result["overlay_source_path"]).name,
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
