-- OnionGuard — ตารางสำหรับเว็บ AI สาธารณะ (แยกจากงานวิจัย)
--
-- รันไฟล์นี้ใน Supabase Dashboard > SQL Editor > New query > วางแล้ว Run
--
-- ทำไมต้องแยกตาราง
-- ตาราง public.scans เก็บหอมแดง 60 หัวของงานวิจัย ทุกแถวมีผล CompactDry YM
-- กำกับ และถูกใช้เทรนโมเดลโดยตรง (src/export_features_from_db.py อ่านจากตารางนี้)
-- ถ้าเว็บสาธารณะเขียนลงตารางเดียวกัน ทุกครั้งที่มีคนกดลองเล่นจะมีแถวขยะปนเข้าไป
-- ในชุดข้อมูลวิจัย — เกิดขึ้นจริงมาแล้ว (แถว 'hhh', 'test', 's002' ตัวพิมพ์เล็ก)
--
-- ตารางนี้จึงไม่มีคอลัมน์ compactdry_truth เลย ไม่ใช่แค่ปล่อยว่าง: สแกน
-- สาธารณะไม่มีทางมีผลเพาะเชื้อมากำกับ การมีช่องนั้นอยู่จะชวนให้เข้าใจผิดว่า
-- ข้อมูลชุดนี้เอาไปเทรนต่อได้
--
-- ข้อจำกัดที่ต้องรู้: นี่คือการแยก "ข้อมูล" ไม่ใช่การแยก "สิทธิ์" ทั้งสอง
-- ตารางเขียนผ่าน service_role ของเซิร์ฟเวอร์เดียวกัน คนที่รู้ URL ของ endpoint
-- ฝั่งวิจัยยังยิงเข้าไปได้อยู่ ถ้าต้องการกันจริงต้องเพิ่มระบบล็อกอินให้หน้า
-- เจ้าหน้าที่ ซึ่งยังไม่ได้ทำ

create table if not exists public.public_scans (
  id                bigint generated always as identity primary key,
  scan_code         text not null,           -- รหัสอัตโนมัติ ผู้ใช้ไม่ต้องกรอก
  captured_at       timestamptz not null default now(),

  image_path        text,                    -- path ใน Storage bucket 'public-scans'

  pred_label        smallint check (pred_label in (0, 1)),
  pred_conf         real check (pred_conf >= 0 and pred_conf <= 1),
  pred_proba        real check (pred_proba >= 0 and pred_proba <= 1),
  decision_threshold real,

  features          jsonb,

  ood_status        text,                    -- in_range / out_of_range / unknown
  borderline        boolean default false,
  framing_ok        boolean,
  quality_notes     jsonb,

  synced_at         timestamptz not null default now()
);

comment on table public.public_scans is
  'สแกนจากเว็บ AI สาธารณะ — ไม่มีผลแล็บกำกับ ห้ามนำไปเทรนโมเดล';
comment on column public.public_scans.scan_code is
  'รหัสที่ระบบสร้างให้อัตโนมัติ ใช้อ้างอิงผลย้อนหลังเท่านั้น ไม่ผูกกับตัวอย่างวิจัย';

create index if not exists public_scans_captured_at_idx
  on public.public_scans (captured_at desc);
create index if not exists public_scans_code_idx
  on public.public_scans (scan_code);

-- ---------------------------------------------------------------- RLS

alter table public.public_scans enable row level security;

-- เว็บ (anon key) อ่านได้อย่างเดียว เหมือนตาราง scans
drop policy if exists "anon can read public_scans" on public.public_scans;
create policy "anon can read public_scans"
  on public.public_scans for select
  to anon
  using (true);

-- การเขียนผ่าน service_role เท่านั้น (ฝั่งเซิร์ฟเวอร์) — ไม่ต้องเขียน policy
-- ให้ service_role เพราะมันข้าม RLS อยู่แล้ว การไม่มี policy insert สำหรับ
-- anon = anon เขียนไม่ได้

-- ---------------------------------------------------------------- Storage

insert into storage.buckets (id, name, public)
values ('public-scans', 'public-scans', false)
on conflict (id) do nothing;

drop policy if exists "service role manages public scan images" on storage.objects;
create policy "service role manages public scan images"
  on storage.objects for all
  to service_role
  using (bucket_id = 'public-scans')
  with check (bucket_id = 'public-scans');
