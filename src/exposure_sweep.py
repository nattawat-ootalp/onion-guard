"""
Exposure/gain sweep for calibrating the UV box.

The correct exposure depends on the box, the UV lamp, and the camera
together, so it cannot be guessed from a datasheet — it has to be measured
against the real rig. This walks a grid of exposure_time_absolute x gain and
reports, for each, how bright the frame comes out and how much detail it
carries.

Reading the output:
  meanR/G/B  average channel level (0-255). A frame near 0 is black; the
             classifier will happily score it and the answer is meaningless.
  max        brightest pixel. If this pins at 255 the highlights are clipped
             and fluorescence intensity is no longer measurable.
  focusVar   variance of the Laplacian. Near 0 means no detail (dark or out
             of focus); higher means real structure is present.

Aim for a frame with real detail and headroom below clipping — not the
brightest one. Then put the chosen pair into config.json -> controls and
re-run capture_lock_test.py to confirm it is stable shot to shot.

Usage:  ./venv/bin/python src/exposure_sweep.py [out_dir]
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera  # noqa: E402

EXPOSURES = [166, 500, 1000, 2500, 5000, 10000]
GAINS = [64, 128]


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg = camera.load_config()
    dev = cfg["camera"]["device_path"]
    width = cfg["camera"]["width"]
    height = cfg["camera"]["height"]
    warmup = cfg["camera"]["warmup_frames"]

    # Check presence up front. "could not open" is ambiguous between the
    # device being missing and it being held by another process (the web app
    # keeps it open for its whole lifetime), and those need opposite fixes.
    if not camera.device_available(cfg):
        print(f"ไม่พบ {dev}")
        print("  - กล้องหลุด/ไม่ได้เสียบ? ตรวจด้วย: lsusb และ v4l2-ctl --list-devices")
        print("  - หรือ node เปลี่ยนเลข? แก้ camera.device_path ใน config.json")
        return
    print(f"device: {dev} (พบแล้ว)\n")

    header = f"{'exposure':>9} {'gain':>5} {'meanR':>7} {'meanG':>7} {'meanB':>7} {'max':>5} {'focusVar':>9}"
    print(header)
    print("-" * len(header))

    for gain in GAINS:
        for exp in EXPOSURES:
            controls = {k: v for k, v in cfg["controls"].items() if not k.startswith("_")}
            controls["exposure_time_absolute"] = exp
            controls["gain"] = gain
            camera.set_v4l2_controls(dev, controls)

            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"{exp:>9} {gain:>5}   เปิดกล้องไม่ได้")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(warmup):
                cap.read()
            ok, frame = cap.read()
            cap.release()

            if not ok or frame is None:
                print(f"{exp:>9} {gain:>5}   อ่านภาพไม่สำเร็จ")
                continue

            b = frame[:, :, 0].mean()
            g = frame[:, :, 1].mean()
            r = frame[:, :, 2].mean()
            lap = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            print(f"{exp:>9} {gain:>5} {r:>7.1f} {g:>7.1f} {b:>7.1f} {frame.max():>5} {lap:>9.1f}")

            if out_dir:
                cv2.imwrite(str(out_dir / f"exp{exp}_gain{gain}.png"), frame)

    if out_dir:
        print(f"\nบันทึกภาพทั้งหมดไว้ที่ {out_dir}")


if __name__ == "__main__":
    main()
