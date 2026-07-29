"""Build data/features.csv from the real scans stored in Supabase.

Every scan row already carries the full feature dict it was predicted from,
computed by the same code path a live scan uses. Re-deriving the numbers from
the stored images would risk a silent mismatch between what the database says
a sample measured and what training sees, so this reads the features straight
out of the table.

Rows with no CompactDry result yet (compactdry_truth is null) are dropped:
null means "not cultured", never "no fungus", so they cannot be training data.

    python src/export_features_from_db.py                  # writes data/features.csv
    python src/export_features_from_db.py --exclude S045   # drop bad framing
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract_features import EXTRA_HEAD_FEATURES, FEATURE_NAMES  # noqa: E402
from supabase_client import SupabaseClient, load_env  # noqa: E402

ID_COL = "sample_code"
LABEL_COL = "compactdry"
OUT_PATH = ROOT / "data" / "features.csv"

# Same column order extract_features.py writes: the per-view feature block,
# then the head-level extras. Anything that ends up here must also be in the
# stored feature dict, or the row is dropped rather than silently zero-filled.
ALL_FEATURES = list(FEATURE_NAMES) + list(EXTRA_HEAD_FEATURES)


def main():
    ap = argparse.ArgumentParser(description="ดึงฟีเจอร์ของสแกนจริงจาก Supabase มาเป็น features.csv")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="รหัสตัวอย่างที่ไม่เอา (เช่น ตัวที่หาหัวหอมไม่เจอ)")
    ap.add_argument("--require-framing-ok", action="store_true",
                    help="ตัดตัวอย่างที่ auto-framing ล้มเหลวออกทั้งหมด")
    args = ap.parse_args()

    load_env()
    client = SupabaseClient()
    rows = client.select(
        "scans",
        columns="sample_code,features,compactdry_truth,framing_ok",
        order="sample_code.asc",
    )

    exclude = set(args.exclude)
    kept, dropped = [], []
    for r in rows:
        code = r["sample_code"]
        if code in exclude:
            dropped.append((code, "ถูกสั่งตัดออก"))
            continue
        if r.get("compactdry_truth") is None:
            dropped.append((code, "ยังไม่มีผลแล็บ"))
            continue
        if args.require_framing_ok and r.get("framing_ok") is False:
            dropped.append((code, "auto-framing ล้มเหลว"))
            continue
        feats = r.get("features") or {}
        missing = [f for f in ALL_FEATURES if f not in feats]
        if missing:
            dropped.append((code, f"ฟีเจอร์ขาด {len(missing)} ตัว"))
            continue
        kept.append(r)

    if not kept:
        sys.exit("ไม่มีแถวที่ใช้ได้เลย")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([ID_COL] + ALL_FEATURES + [LABEL_COL])
        for r in kept:
            feats = r["features"]
            writer.writerow([r["sample_code"]]
                            + [feats[f] for f in ALL_FEATURES]
                            + [int(r["compactdry_truth"])])

    labels = [int(r["compactdry_truth"]) for r in kept]
    print(f"เขียน {out} — {len(kept)} แถว, {len(ALL_FEATURES)} ฟีเจอร์")
    print(f"  พบ (1) = {labels.count(1)} | ไม่พบ (0) = {labels.count(0)}")
    if dropped:
        print(f"  ตัดออก {len(dropped)}: " + ", ".join(f"{c} ({why})" for c, why in dropped))


if __name__ == "__main__":
    main()
