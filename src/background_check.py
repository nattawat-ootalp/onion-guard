"""
Background material check for the UV box.

Run this with the box set up as usual but NO onion in it, to judge whether
the background surface is suitable before spending time on exposure
calibration or data collection.

Two things disqualify a background, and they pull in opposite directions —
which is why this is worth measuring rather than eyeballing:

  FLUORESCENCE  White paper, most fabrics and many "black" materials are
                bleached with optical brighteners, which are designed to
                glow under UV. A glowing background is far brighter than
                the onion and its veiling glare in the lens washes out the
                faint fluorescing spots that the classifier looks for.

  SPECULAR REFLECTION  A glossy surface mirrors the UV tube straight into
                the lens as a bright band. Covering it with paper hides the
                band but introduces the fluorescence problem above; the
                real fix is a MATTE, NON-fluorescent surface (or tilting
                the base so the reflection misses the lens).

The band detector looks for a row (or column) far brighter than the frame
median, which is what a mirrored tube looks like. A diffuse glow raises the
whole frame instead and shows up as a high overall mean.

The check deliberately runs at HIGH SENSITIVITY (see PROBE_EXPOSURE/
PROBE_GAIN), not at the exposure in config.json. The question here is "does
this material glow at all?", so the test wants maximum sensitivity — and an
under-exposed frame would pass both checks simply by being black, which is
false reassurance rather than a result. A frame too dark to judge is
reported as inconclusive instead of as a pass.

Usage:  ./venv/bin/python src/background_check.py [out.png] [exposure] [gain]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera  # noqa: E402

# A background this bright is competing with the sample rather than sitting
# behind it. Onion fluorescence itself is faint, so the floor has to be low.
MEAN_GOOD = 8.0
MEAN_WARN = 20.0
# A row/column this many times the frame median reads as a mirrored tube
# rather than an evenly lit surface.
BAND_RATIO = 3.0

# High-sensitivity probe settings, used instead of config.json's exposure.
# A faint glow is exactly what must be detected, so the test runs the sensor
# wide open rather than at the (possibly uncalibrated) working exposure.
PROBE_EXPOSURE = 10000
PROBE_GAIN = 128

# Below this the frame carries no usable information and neither check means
# anything — reported as inconclusive, never as a pass.
TOO_DARK_MAX = 40
TOO_DARK_MEAN = 1.0


def find_band(gray, axis):
    """Return (index, mean, ratio_vs_median) of the brightest row/column."""
    profile = gray.mean(axis=axis)
    idx = int(profile.argmax())
    med = float(np.median(profile))
    ratio = profile[idx] / med if med > 1e-6 else float("inf")
    return idx, float(profile[idx]), ratio


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    exposure = int(sys.argv[2]) if len(sys.argv) > 2 else PROBE_EXPOSURE
    gain = int(sys.argv[3]) if len(sys.argv) > 3 else PROBE_GAIN

    cfg = camera.load_config()
    if not camera.device_available(cfg):
        print(f"ไม่พบ {cfg['camera']['device_path']} — ตรวจ lsusb / v4l2-ctl --list-devices")
        return 1

    # Probe at high sensitivity rather than the config exposure — see docstring.
    cfg["controls"] = {k: v for k, v in cfg["controls"].items() if not k.startswith("_")}
    cfg["controls"]["exposure_time_absolute"] = exposure
    cfg["controls"]["gain"] = gain
    print(f"probe settings: exposure={exposure} gain={gain} (ไวสูงสุด เพื่อจับการเรืองแสงจาง ๆ)")

    cap = camera.open_camera(cfg)
    frame = camera.capture_frame(cap)
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, g, r = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
    mean = float(gray.mean())

    print(f"frame        : {frame.shape[1]}x{frame.shape[0]}")
    print(f"mean level   : {mean:.1f}   (R={r:.1f} G={g:.1f} B={b:.1f})")
    print(f"max / clipped: {gray.max()} / {(frame.max(axis=2) >= 255).mean() * 100:.3f}%")

    # A near-black frame would sail through both checks below, so refuse to
    # report a verdict on one.
    if gray.max() < TOO_DARK_MAX and mean < TOO_DARK_MEAN:
        print("\n--- สรุปไม่ได้ ---")
        print(f"  ภาพมืดเกินกว่าจะตัดสิน (mean {mean:.1f}, max {gray.max()})")
        print("  แม้เปิดไวสูงสุดแล้วยังไม่มีสัญญาณ แปลว่า:")
        print("   - ไฟ UV ปิดอยู่? หรือส่องไม่ถึงพื้นที่ที่กล้องเห็น?")
        print("   - ฝาปิดบังเลนส์ / กล้องหันผิดทาง?")
        print("  *ยังไม่ถือว่าพื้นหลังผ่าน* — ต้องเห็นภาพก่อนถึงจะบอกได้ว่าวัสดุใช้ได้ไหม")
        if out_path:
            cv2.imwrite(out_path, frame)
            cv2.imwrite(str(Path(out_path).with_name(Path(out_path).stem + "_boost.png")),
                        cv2.convertScaleAbs(frame, alpha=6.0))
            print(f"\n  บันทึก {out_path} (+_boost) ไว้ดูแล้ว")
        return 2

    print("\n--- 1) พื้นหลังเรืองแสงไหม ---")
    if mean <= MEAN_GOOD:
        print(f"  ผ่าน  — mean {mean:.1f} <= {MEAN_GOOD} พื้นหลังมืดพอ ไม่เรืองแสงกวน")
    elif mean <= MEAN_WARN:
        print(f"  ก้ำกึ่ง — mean {mean:.1f} สว่างกว่าที่ควร ยังพอใช้ได้แต่ contrast จะลด")
    else:
        print(f"  ไม่ผ่าน — mean {mean:.1f} > {MEAN_WARN} พื้นหลังเรืองแสงแรง")
        print("          วัสดุน่าจะมี optical brightener (กระดาษ/ผ้าฟอกขาว)")
        print("          เปลี่ยนเป็นดำด้านที่ไม่เรืองแสง")

    print("\n--- 2) มีเงาหลอดสะท้อนไหม ---")
    row_i, row_v, row_ratio = find_band(gray, axis=1)
    col_i, col_v, col_ratio = find_band(gray, axis=0)
    worst = max(row_ratio, col_ratio)

    if row_ratio >= col_ratio:
        where, idx, val, ratio = "แถวแนวนอน y", row_i, row_v, row_ratio
    else:
        where, idx, val, ratio = "คอลัมน์แนวตั้ง x", col_i, col_v, col_ratio

    if worst >= BAND_RATIO:
        band = frame[max(0, row_i - 8):row_i + 8] if row_ratio >= col_ratio else frame[:, max(0, col_i - 8):col_i + 8]
        bb, bg, br = band[:, :, 0].mean(), band[:, :, 1].mean(), band[:, :, 2].mean()
        print(f"  ไม่ผ่าน — เจอแถบสว่างที่ {where}={idx} (สว่าง {val:.1f}, {ratio:.1f} เท่าของค่ากลาง)")
        print(f"          สีแถบ B={bb:.0f} G={bg:.0f} R={br:.0f}"
              f"{'  (B เด่น = ตัวหลอด/เงาหลอดโดยตรง)' if bb > br * 1.5 else ''}")
        print("          แก้: ใช้พื้นด้านแทนพื้นมัน / เอียงแผ่นรอง 10-15 องศา / บังหลอดให้พ้นเลนส์")
    else:
        print(f"  ผ่าน  — ไม่พบแถบสว่างผิดปกติ (สูงสุด {worst:.1f} เท่า < {BAND_RATIO})")

    print("\n--- สรุป ---")
    ok = mean <= MEAN_WARN and worst < BAND_RATIO
    print("  พื้นหลังใช้ได้ ไปตั้ง exposure ต่อได้ (src/exposure_sweep.py)" if ok
          else "  ยังไม่ผ่าน แก้ตามข้างบนแล้วรันซ้ำ")

    if out_path:
        cv2.imwrite(out_path, frame)
        # Brightened copy, because a correct background is too dark to judge
        # on screen by eye.
        boost = str(Path(out_path).with_name(Path(out_path).stem + "_boost.png"))
        cv2.imwrite(boost, cv2.convertScaleAbs(frame, alpha=6.0))
        print(f"\n  บันทึก {out_path} และ {boost} (เร่งความสว่าง 6 เท่าให้ดูด้วยตาได้)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
