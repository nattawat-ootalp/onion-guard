"""
Freeze the garlic "normal" baseline into models/garlic_anomaly.json.

Every garlic clove photographed so far is a normal clove — that is the whole
training set, and it is one class (see garlic_anomaly.py). This script reads
those cloves' stored feature dicts, fits the baseline, and writes it to disk
so the deployed detector is a file in git rather than something refitted at
runtime on whatever happens to be in the database that day.

    python src/train_garlic_anomaly.py                     # from Supabase
    python src/train_garlic_anomaly.py --from-csv f.csv    # from a features.csv
    python src/train_garlic_anomaly.py --exclude GA014     # drop a bad clove

WHICH ROWS COUNT AS NORMAL
Rows that already carry a pred_label were scanned by this detector, so some
of them may be the anomalies it found; folding them back in would teach the
baseline that anomalies are normal. They are dropped unless --include-scored
is given. A row with a CompactDry result of 1 (fungus confirmed) is dropped
always — whatever else it is, it is not a normal clove.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import garlic_anomaly as ga  # noqa: E402


def load_config():
    import json
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def rows_from_supabase(args):
    from supabase_client import SupabaseClient, load_env
    load_env()
    client = SupabaseClient()
    rows = client.select(
        "scans",
        columns="sample_code,crop,features,pred_label,compactdry_truth,framing_ok",
        order="sample_code.asc",
        filters={"crop": f"eq.{args.crop}"},
    )
    kept, dropped = [], []
    for r in rows:
        code = r.get("sample_code") or "?"
        if code in set(args.exclude):
            dropped.append((code, "ถูกสั่งตัดออก"))
        elif not r.get("features"):
            dropped.append((code, "ไม่มีฟีเจอร์ที่เก็บไว้"))
        elif r.get("compactdry_truth") == 1:
            dropped.append((code, "ผลแล็บพบเชื้อ — ไม่ใช่ตัวอย่างปกติ"))
        elif r.get("pred_label") is not None and not args.include_scored:
            dropped.append((code, "เคยถูกตรวจด้วยตัวตรวจนี้แล้ว"))
        elif args.require_framing_ok and r.get("framing_ok") is False:
            dropped.append((code, "auto-framing ล้มเหลว"))
        else:
            kept.append((code, r["features"]))
    return kept, dropped, f"supabase:scans?crop={args.crop}"


def rows_from_csv(path, args):
    kept, dropped = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r.get("sample_code") or "?"
            if code in set(args.exclude):
                dropped.append((code, "ถูกสั่งตัดออก"))
                continue
            feats = {}
            for k, v in r.items():
                try:
                    feats[k] = float(v)
                except (TypeError, ValueError):
                    continue
            kept.append((code, feats))
    return kept, dropped, f"csv:{Path(path).name}"


def main():
    ap = argparse.ArgumentParser(
        description="สร้างค่าอ้างอิง “กระเทียมปกติ” สำหรับตัวตรวจความผิดปกติ")
    ap.add_argument("--crop", default="garlic", choices=["garlic", "onion"])
    ap.add_argument("--from-csv", default=None,
                    help="อ่านฟีเจอร์จากไฟล์ CSV แทนการดึงจาก Supabase")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="รหัสตัวอย่างที่ไม่เอา (เช่น ภาพที่ครอบพลาด)")
    ap.add_argument("--include-scored", action="store_true",
                    help="รวมแถวที่เคยถูกตรวจด้วยตัวตรวจนี้แล้วด้วย (ปกติจะตัดออก)")
    ap.add_argument("--require-framing-ok", action="store_true")
    ap.add_argument("--out", default=str(ga.BASELINE_PATH))
    args = ap.parse_args()

    if args.from_csv:
        kept, dropped, source = rows_from_csv(args.from_csv, args)
    else:
        kept, dropped, source = rows_from_supabase(args)

    print(f"ใช้ได้ {len(kept)} ตัวอย่าง · ตัดออก {len(dropped)} ตัวอย่าง")
    for code, why in dropped:
        print(f"  - {code}: {why}")
    if not kept:
        sys.exit("ไม่มีตัวอย่างที่ใช้ได้เลย")

    cfg = load_config()
    baseline = ga.fit([f for _code, f in kept], cfg=cfg, source=source)
    baseline["sample_codes"] = [code for code, _f in kept]

    out = ga.save_baseline(baseline, args.out)
    print(f"\nเขียน {out} แล้ว")
    print(f"  ฟีเจอร์ที่ใช้ {len(baseline['features'])} ตัว")
    print(f"  เกณฑ์ตัดสิน {baseline['threshold']:.2f}")
    ts = baseline["train_scores"]
    print(f"  คะแนนของตัวอย่างปกติ: กลาง {ts['p50']:.2f} · p90 {ts['p90']:.2f} "
          f"· สูงสุด {ts['max']:.2f}")
    print(f"  ตัวอย่างปกติที่จะถูกแจ้งว่าผิดปกติ (false alarm ในชุดฝึก): "
          f"{baseline['train_flagged']}/{baseline['n_samples']}")


if __name__ == "__main__":
    main()
