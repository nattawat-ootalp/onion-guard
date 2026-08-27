"""Batch-ingest a folder of real photos through the running web app.

Folder layout expected (sample code = folder name):

    <root>/S001/UV/<one photo>
    <root>/S001/visible_light/<one photo>     (optional)
    <root>/S002/UV/...

Why it drives the HTTP API instead of calling predict_head directly: the
phone-upload path does more than predict — EXIF orientation, auto-framing,
the quality checks, saving the frame, and the Supabase insert all live in
web/app.py. Re-implementing any of that here would create a second, silently
divergent pipeline, and the whole point of these rows is that they were
produced exactly the way a real scan is.

Run the server first (it must have Supabase credentials in .env):

    python web/app.py
    python src/ingest_folder.py --root /mnt/c/Users/.../onion

Nothing is deleted or overwritten: each run appends new scan rows. Use
--skip-existing to leave sample codes that are already in the database alone.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UV_DIR = "UV"
VISIBLE_DIR = "visible_light"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".dng", ".heic", ".heif"}


def _first_image(folder):
    """The single photo in a step folder, or None.

    More than one file is not an error worth stopping for — the protocol is
    one shot per light condition, so an extra file is almost always a stray
    (a duplicate export, a .thumb). Taking the alphabetically first keeps the
    run deterministic and the count is reported so it can be checked.
    """
    if not folder.is_dir():
        return None, 0
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return (files[0] if files else None), len(files)


def _post_multipart(url, fields, files, timeout):
    """Minimal multipart/form-data POST — no requests dependency.

    src/predict.py's minimal-dependency contract is about the Pi, not this
    script, but there is no reason to add a package for one call.
    """
    boundary = "----onionguard" + str(int(time.time() * 1000))
    body = bytearray()
    for key, value in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                 f"{value}\r\n").encode()
    for key, path in files:
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{key}"; filename="{path.name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode()
        body += path.read_bytes()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _error_body(exc):
    try:
        return json.loads(exc.read().decode()).get("error", str(exc))
    except Exception:  # noqa: BLE001
        return str(exc)


def ingest_sample(api, sample_code, uv_path, visible_path, timeout, crop="onion"):
    """Upload one head and run it, returning the server's predict payload.

    crop travels with the upload because the server pins it to the session:
    it decides which table column the row carries and, for a crop with no
    model, whether anything is classified at all."""
    files = [("images", uv_path)]
    if visible_path is not None:
        files.append(("images", visible_path))

    cap = _post_multipart(f"{api}/capture",
                          {"sample_code": sample_code, "crop": crop}, files, timeout)
    session_id = cap["session_id"]

    # No visible-light photo: the step is optional, so tell the server to skip
    # it rather than leaving the session one step short of complete.
    if not cap.get("all_captured"):
        _post_json(f"{api}/capture/skip", {"session_id": session_id}, timeout)

    result = _post_json(f"{api}/predict-session", {"session_id": session_id}, timeout)
    result["_upload_warnings"] = [w for a in cap.get("accepted", []) for w in a.get("warnings", [])]
    return result


def existing_sample_codes(api, timeout):
    """Sample codes already in the database, via the app's own read path."""
    with urllib.request.urlopen(f"{api}/api/samples", timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    rows = data.get("samples", data) if isinstance(data, dict) else data
    return {r.get("sample_code") for r in rows if isinstance(r, dict)}


def main():
    ap = argparse.ArgumentParser(description="อัปโหลดภาพจริงทั้งโฟลเดอร์เข้าระบบ")
    ap.add_argument("--root", required=True, help="โฟลเดอร์แม่ที่มีโฟลเดอร์รหัสตัวอย่างข้างใน")
    ap.add_argument("--api", default="http://127.0.0.1:5000", help="URL ของเซิร์ฟเวอร์")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="ข้ามรหัสตัวอย่างที่มีในฐานข้อมูลแล้ว (กันข้อมูลซ้ำ)")
    ap.add_argument("--dry-run", action="store_true", help="แสดงว่าจะทำอะไรบ้าง ไม่ส่งจริง")
    ap.add_argument("--crop", default="onion", choices=["onion", "garlic"],
                    help="ชนิดพืชของทั้งโฟลเดอร์ (ค่าเริ่มต้น onion) — กระเทียมยังไม่มีโมเดล "
                         "ระบบจะเก็บภาพและค่าที่วัดได้อย่างเดียว")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"ไม่พบโฟลเดอร์: {root}")

    api = args.api.rstrip("/")
    skip_codes = set()
    if args.skip_existing and not args.dry_run:
        try:
            skip_codes = existing_sample_codes(api, args.timeout)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"อ่านรายการตัวอย่างเดิมไม่ได้: {exc}")

    jobs, empty = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        uv_path, uv_n = _first_image(folder / UV_DIR)
        vis_path, vis_n = _first_image(folder / VISIBLE_DIR)
        if uv_path is None:
            empty.append(folder.name)
            continue
        jobs.append((folder.name, uv_path, vis_path, uv_n, vis_n))

    print(f"พบโฟลเดอร์ที่มีภาพ UV: {len(jobs)} | ไม่มีภาพ ข้ามไป: {len(empty)}")
    if empty:
        print("  ข้าม: " + ", ".join(empty))
    for code, _u, vis, uv_n, vis_n in jobs:
        extra = []
        if uv_n > 1:
            extra.append(f"UV มี {uv_n} ไฟล์ ใช้ไฟล์แรก")
        if vis_n > 1:
            extra.append(f"visible มี {vis_n} ไฟล์ ใช้ไฟล์แรก")
        if vis is None:
            extra.append("ไม่มีภาพแสงปกติ")
        if extra:
            print(f"  {code}: " + "; ".join(extra))

    if args.dry_run:
        return

    ok, failed, saved_ok = 0, 0, 0
    print()
    for i, (code, uv_path, vis_path, _un, _vn) in enumerate(jobs, 1):
        if code in skip_codes:
            print(f"[{i}/{len(jobs)}] {code}: มีในฐานข้อมูลแล้ว ข้าม")
            continue
        started = time.time()
        try:
            res = ingest_sample(api, code, uv_path, vis_path, args.timeout, crop=args.crop)
        except urllib.error.HTTPError as exc:
            failed += 1
            print(f"[{i}/{len(jobs)}] {code}: ล้มเหลว — {_error_body(exc)}")
            continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(jobs)}] {code}: ล้มเหลว — {exc}")
            continue

        ok += 1
        db = res.get("saved_to_db") or {}
        if db.get("ok"):
            saved_ok += 1
        flags = []
        if res.get("borderline"):
            flags.append("ก้ำกึ่ง")
        if (res.get("ood_check") or {}).get("status") == "out_of_range":
            flags.append("นอกช่วงข้อมูลเทรน")
        if res.get("framing_failed_views"):
            flags.append("หาหัวหอมไม่เจอ")
        flags += res.get("_upload_warnings", [])
        if not db.get("ok"):
            flags.append(f"บันทึกลงฐานข้อมูลไม่สำเร็จ: {db.get('reason')}")

        if res.get("collect_only"):
            summary = (f"เก็บข้อมูลอย่างเดียว (ไม่มีโมเดล{res.get('crop_label', '')}) "
                       f"| ฟีเจอร์ {res.get('n_features')} ค่า "
                       f"| จุดเล็ก {res.get('n_small_blobs')}")
        elif res.get("screening") == "anomaly":
            # The one-class path reports a distance from normal, not a class
            # probability — printing p= here would print None on every clove.
            an = res.get("anomaly") or {}
            summary = (f"{res.get('label_text')} ({res.get('confidence_pct')}% "
                       f"| ความต่างจากปกติ {an.get('score')} เกณฑ์ {an.get('threshold')} "
                       f"| จุดเล็ก {res.get('n_small_blobs')})")
        else:
            summary = (f"{res.get('label_text')} ({res.get('confidence_pct')}% "
                       f"| p={res.get('proba_positive')} "
                       f"| จุดเล็ก {res.get('n_small_blobs')})")
        print(f"[{i}/{len(jobs)}] {code}: {summary} {time.time() - started:.1f}s"
              + ("  ⚠ " + " / ".join(flags) if flags else ""))

    print(f"\nสำเร็จ {ok} | ล้มเหลว {failed} | บันทึกลงฐานข้อมูล {saved_ok}")


if __name__ == "__main__":
    main()
