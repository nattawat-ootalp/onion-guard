"""
Camera capture for the Pi, with a mock fallback for dev machines.

CONTROL LOCKING GOES THROUGH v4l2-ctl, NOT cv2's cap.set()
UV fluorescence brightness is the measurement itself, so if the camera
re-exposes or re-white-balances between shots the same onion yields
different features. OpenCV's cap.set(CAP_PROP_*) silently fails on many
V4L2 drivers, so controls are applied by shelling out to v4l2-ctl and then
READ BACK to prove they stuck. Anything that did not stick is reported in
unlocked_settings and surfaced in the UI — an unlocked exposure invalidates
a dataset quietly, which is the worst way to lose one.

ORDER MATTERS. A manual control is marked "inactive" by the driver while
its auto counterpart is on, and writes to it are discarded. So every auto
switch (auto_exposure, white_balance_automatic, focus_automatic_continuous,
exposure_dynamic_framerate) is applied FIRST, in _AUTO_FIRST order, before
the manual values that depend on them.

TWO FRAME CONTRACTS, DELIBERATELY
  capture_frame(cap)      -> RAW frame, native resolution.
                             Used by calibrate.py, which draws a pixel grid
                             to pick ROI coordinates — those coordinates are
                             only meaningful in the raw frame.
  OpenCVCamera.capture()  -> centre-cropped square, resized to
                             config image.size_px.
                             The blob detector's thresholds
                             (min_blob_area_px, small_area_max_px,
                             large_area_min_px, fine/coarse filter sizes)
                             are absolute pixel counts calibrated at
                             640x640. A raw 1280x720 frame would make every
                             one of them mean a different physical size, and
                             the features would not be comparable to the
                             training set.

load_config / open_camera / capture_frame keep the signatures the existing
calibrate.py and capture_lock_test.py import, so those keep working.
"""
import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

# Applied before anything else: while these are on, the driver marks the
# corresponding manual control "inactive" and ignores writes to it.
_AUTO_FIRST = [
    "auto_exposure",
    "white_balance_automatic",
    "focus_automatic_continuous",
    "exposure_dynamic_framerate",
]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- v4l2 controls

def _ordered_controls(controls):
    """Auto switches first (see _AUTO_FIRST), then everything else."""
    items = [(k, v) for k, v in controls.items() if k != "note" and not k.startswith("_")]
    autos = [(k, v) for k, v in items if k in _AUTO_FIRST]
    autos.sort(key=lambda kv: _AUTO_FIRST.index(kv[0]))
    rest = [(k, v) for k, v in items if k not in _AUTO_FIRST]
    return autos + rest


def set_v4l2_controls(device_path, controls):
    """Apply controls via v4l2-ctl. Returns a list of (name, error) failures."""
    failures = []
    if shutil.which("v4l2-ctl") is None:
        return [("v4l2-ctl", "ไม่ได้ติดตั้ง v4l2-ctl (apt install v4l-utils)")]

    for name, value in _ordered_controls(controls):
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "-c", f"{name}={value}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            failures.append((name, (result.stderr or result.stdout).strip()))
    return failures


def read_v4l2_controls(device_path, names):
    """Read controls back. Returns {name: int_value} for whatever could be read.

    Menu controls print their label alongside the number
    ("auto_exposure: 1 (Manual Mode)"), so only the leading integer is taken.
    Parsing the whole field would drop every menu control and report it as
    unlocked — a correctly locked rig flagged as broken.
    """
    out = {}
    if shutil.which("v4l2-ctl") is None or not names:
        return out
    result = subprocess.run(
        ["v4l2-ctl", "-d", device_path, "--get-ctrl", ",".join(names)],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        token = val.strip().split()[0] if val.strip() else ""
        try:
            out[key.strip()] = int(token)
        except ValueError:
            pass
    return out


def verify_locks(device_path, controls):
    """Compare requested vs actual. Returns (actual, mismatched_names).

    A name in mismatched_names is NOT locked — the driver kept its own value,
    so that setting can still drift between shots.
    """
    requested = {k: v for k, v in _ordered_controls(controls)}
    actual = read_v4l2_controls(device_path, list(requested))
    mismatched = [
        name for name, want in requested.items()
        if name not in actual or int(actual[name]) != int(want)
    ]
    return actual, mismatched


# ---------------------------------------------------------------- geometry

def normalize_frame(frame_bgr, target_size):
    """Centre-crop to square, then resize to target_size — see module docstring."""
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    square = frame_bgr[top:top + side, left:left + side]
    if side != target_size:
        interp = cv2.INTER_AREA if side > target_size else cv2.INTER_LINEAR
        square = cv2.resize(square, (target_size, target_size), interpolation=interp)
    return square


def device_available(cfg):
    """Whether the CONFIGURED device exists. Deliberately not a /dev/video*
    glob: a Pi exposes many video nodes that are codec/ISP devices, not
    cameras, so their presence proves nothing about the actual camera."""
    return Path(cfg["camera"]["device_path"]).exists()


# ---------------------------------------------------------------- open / read
# Signatures below are what calibrate.py and capture_lock_test.py import.

def open_camera(cfg=None, warmup_frames=None):
    """Open the configured camera, lock its controls, burn the warm-up frames,
    and return a ready-to-read cv2.VideoCapture (raw resolution)."""
    if cfg is None:
        cfg = load_config()
    cam = cfg["camera"]
    device_path = cam["device_path"]
    if warmup_frames is None:
        warmup_frames = cam["warmup_frames"]

    failures = set_v4l2_controls(device_path, cfg.get("controls", {}))
    for name, err in failures:
        print(f"WARNING: failed to set {name}: {err}")

    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดกล้องไม่ได้: {device_path}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.get("fourcc", "YUYV")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

    # Sensors need several frames after a settings change before the output
    # actually reflects them.
    for _ in range(warmup_frames):
        cap.read()

    return cap


def capture_frame(cap):
    """One RAW frame at native resolution (calibration coordinate space)."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("อ่านภาพจากกล้องไม่สำเร็จ")
    return frame


# ---------------------------------------------------------------- backends

class CaptureBackend(ABC):
    @abstractmethod
    def capture(self):
        """Return one BGR frame, already normalized to the training geometry."""

    @abstractmethod
    def describe(self):
        """Dict describing the backend — surfaced in the UI so it is always
        obvious whether a result came from a real camera or from mock data."""

    def close(self):
        pass


class OpenCVCamera(CaptureBackend):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device_path = cfg["camera"]["device_path"]
        self.target_size = cfg["image"]["size_px"]
        self.requested = {k: v for k, v in _ordered_controls(cfg.get("controls", {}))}

        self.cap = open_camera(cfg)
        self.actual, self.unlocked = verify_locks(self.device_path, cfg.get("controls", {}))

    def capture(self):
        return normalize_frame(capture_frame(self.cap), self.target_size)

    def describe(self):
        return {
            "backend": "camera",
            "is_mock": False,
            "device": self.device_path,
            "requested": self.requested,
            "actual": self.actual,
            # Anything here was refused by the driver and is free to drift.
            "unlocked_settings": self.unlocked,
        }

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MockCamera(CaptureBackend):
    """Serves existing mock images so the app runs with no camera attached.

    Cycles through a shuffled pool of real mock samples rather than always
    returning the same picture, so a dev run exercises both the positive and
    the negative result path.
    """

    def __init__(self, cfg, images_dir=None):
        self.cfg = cfg
        self.target_size = cfg["image"]["size_px"]
        self.images_dir = Path(images_dir or (ROOT / "images"))
        self.pool = sorted(self.images_dir.glob("*.png"))
        if not self.pool:
            raise RuntimeError(f"ไม่พบภาพ mock ใน {self.images_dir}")
        # Deterministic ordering seeded from config, so a dev session is
        # reproducible; a time-based pick would make bug reports unrepeatable.
        self._rng = np.random.default_rng(cfg.get("random_seed", 0))
        self._order = list(self._rng.permutation(len(self.pool)))
        self._cursor = 0
        self.last_source = None

    def capture(self):
        idx = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        path = self.pool[idx]
        self.last_source = path.name
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"อ่านภาพ mock ไม่สำเร็จ: {path}")
        # Already 640x640, but normalize anyway so this path behaves
        # identically to the camera path.
        return normalize_frame(frame, self.target_size)

    def describe(self):
        return {
            "backend": "mock",
            "is_mock": True,
            "device": None,
            "pool_size": len(self.pool),
            "last_source": self.last_source,
        }


def pick_backend(cfg, force_mock=False):
    """Real camera if the configured device can actually be opened, else mock.

    Returns (backend, reason) — reason explains the choice so the UI can state
    plainly why it is in mock mode instead of leaving the user to guess
    whether the camera silently failed.
    """
    if force_mock:
        return MockCamera(cfg), "บังคับใช้โหมดภาพจำลอง"

    device_path = cfg["camera"]["device_path"]
    if device_available(cfg):
        try:
            cam = OpenCVCamera(cfg)
            reason = f"ใช้กล้องจริง ({device_path})"
            if cam.unlocked:
                reason += f" — เตือน: ล็อกไม่ติด {len(cam.unlocked)} ค่า"
            return cam, reason
        except Exception as exc:  # noqa: BLE001 - fall back rather than break the app
            camera_error = f"พบ {device_path} แต่เปิดกล้องไม่สำเร็จ ({exc})"
    else:
        camera_error = f"ไม่พบอุปกรณ์กล้องตามที่ตั้งไว้ ({device_path})"

    # The mock fallback needs images/, which is NOT deployed to the Pi (it is
    # dev-only mock data). If it is missing, report the CAMERA failure as the
    # cause — it is the real problem — rather than letting the fallback's own
    # "no mock images" error surface and hide it.
    try:
        return MockCamera(cfg), f"{camera_error} จึงใช้ภาพจำลองแทน"
    except Exception as mock_exc:  # noqa: BLE001
        raise RuntimeError(
            f"{camera_error} และไม่มีภาพจำลองสำรองบนเครื่องนี้ ({mock_exc}) "
            "— แก้ปัญหากล้องก่อน หรือคัดลอกโฟลเดอร์ images/ มาด้วยหากต้องการโหมดจำลอง"
        ) from None


if __name__ == "__main__":
    # Rig check: python src/camera.py [out.png]
    import sys

    cfg = load_config()
    print(f"device_path : {cfg['camera']['device_path']}")
    print(f"exists      : {device_available(cfg)}")

    backend, reason = pick_backend(cfg)
    print(f"backend     : {reason}\n")

    info = backend.describe()
    if not info["is_mock"]:
        req, act = info["requested"], info["actual"]
        print(f"{'control':32s} {'requested':>10s} {'actual':>10s}   status")
        for name in req:
            a = act.get(name, "—")
            ok = name not in info["unlocked_settings"]
            print(f"{name:32s} {str(req[name]):>10s} {str(a):>10s}   {'locked' if ok else 'NOT LOCKED'}")
        if info["unlocked_settings"]:
            print(f"\nWARNING: ไม่ถูกล็อก {len(info['unlocked_settings'])} ค่า: "
                  f"{', '.join(info['unlocked_settings'])}")
        else:
            print("\nล็อกครบทุกค่า")
    else:
        print(json.dumps(info, ensure_ascii=False, indent=2))

    t0 = time.time()
    frame = backend.capture()
    print(f"\ncaptured {frame.shape} in {time.time() - t0:.3f}s")

    out = sys.argv[1] if len(sys.argv) > 1 else "capture_test.png"
    cv2.imwrite(out, frame)
    print(f"wrote {out}")
    backend.close()
