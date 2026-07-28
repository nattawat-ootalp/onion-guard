"""
Compatibility shim for the Pi's project root.

The camera implementation now lives in src/camera.py (one module, v4l2-ctl
based). calibrate.py and capture_lock_test.py sit in the project root and do
`from camera import load_config, open_camera, capture_frame`, so this file is
deployed to the Pi as ./camera.py and re-exports those names.

It loads src/camera.py by explicit FILE PATH rather than `import camera`,
because the root directory is on sys.path when those scripts run — a plain
import would resolve back to this shim and recurse.

Deployed as:  ~/onionguard/camera.py
"""
import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src" / "camera.py"

_spec = importlib.util.spec_from_file_location("onionguard_camera_impl", _SRC)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

# The three names the existing root scripts import.
load_config = _impl.load_config
open_camera = _impl.open_camera
capture_frame = _impl.capture_frame

# The rest of the module, for anything that wants the newer API.
normalize_frame = _impl.normalize_frame
device_available = _impl.device_available
set_v4l2_controls = _impl.set_v4l2_controls
read_v4l2_controls = _impl.read_v4l2_controls
verify_locks = _impl.verify_locks
pick_backend = _impl.pick_backend
CaptureBackend = _impl.CaptureBackend
OpenCVCamera = _impl.OpenCVCamera
MockCamera = _impl.MockCamera

CONFIG_PATH = _impl.CONFIG_PATH
ROOT = _impl.ROOT
