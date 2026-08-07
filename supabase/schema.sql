-- OnionGuard — Supabase schema (เฟส 5)
--
-- รันไฟล์นี้ใน Supabase Dashboard > SQL Editor > New query > วางแล้ว Run
-- (Supabase ไม่เปิดให้รัน DDL ผ่าน REST API ด้วยเหตุผลด้านความปลอดภัย)
--
-- หมายเหตุสำคัญ: สเปกเดิมเขียนไว้ว่าเก็บภาพ 4 มุม (image_urls) แต่โครงงาน
-- เปลี่ยนมาถ่าย "มุมเดียวต่อหัว" แล้ว ตารางนี้จึงเก็บภาพเดียว ถ้าอนาคต
-- กลับไปหลายมุม ให้เพิ่มตาราง scan_images แบบ 1-to-many แทนการยัดเป็น array
-- เพราะจะ query และตั้ง RLS ได้ตรงกว่า

-- ---------------------------------------------------------------- ตารางหลัก

create table if not exists public.scans (
  id                bigint generated always as identity primary key,
  sample_code       text not null,
  captured_at       timestamptz not null default now(),

  -- ภาพ
  image_path        text,                    -- path ใน Storage bucket 'scans'
  image_original    text,                    -- ไฟล์ต้นฉบับที่ผู้ใช้อัปโหลด (ถ้าเก็บ)

  -- ผลจากโมเดล
  pred_label        smallint check (pred_label in (0, 1)),
  pred_conf         real check (pred_conf >= 0 and pred_conf <= 1),
  pred_proba        real check (pred_proba >= 0 and pred_proba <= 1),
  decision_threshold real,

  -- ฟีเจอร์ทั้งชุดเก็บเป็น jsonb: จำนวนฟีเจอร์เปลี่ยนได้ตามการทดลอง
  -- (เคยเป็น 46 ตอนถ่าย 4 มุม ตอนนี้ 23) ถ้าทำเป็นคอลัมน์จะต้อง migrate ทุกครั้ง
  features          jsonb,

  -- ธงคุณภาพจากระบบ เก็บไว้เพื่อให้ย้อนดูได้ว่าผลนั้นเชื่อได้แค่ไหน
  ood_status        text,                    -- in_range / out_of_range / unknown
  borderline        boolean default false,
  framing_ok        boolean,
  quality_notes     jsonb,

  -- ผลตรวจอ้างอิงจากห้องแล็บ (กรอกทีหลังหลังเพาะเชื้อเสร็จ)
  -- null = ยังไม่มีผล ไม่ใช่ "ไม่พบเชื้อ"
  compactdry_truth  smallint check (compactdry_truth in (0, 1)),
  truth_recorded_at timestamptz,

  note              text,
  synced_at         timestamptz not null default now()
);

comment on column public.scans.compactdry_truth is
  'ผล CompactDry YM: 1=พบ, 0=ไม่พบ, null=ยังไม่ได้ตรวจ — null ไม่ได้แปลว่าไม่พบ';
comment on column public.scans.features is
  'ฟีเจอร์ทั้งชุดที่ใช้ทำนายครั้งนั้น เก็บไว้เพื่อเทรนซ้ำได้โดยไม่ต้องประมวลผลภาพใหม่';

create index if not exists scans_sample_code_idx on public.scans (sample_code);
create index if not exists scans_captured_at_idx on public.scans (captured_at desc);
create index if not exists scans_truth_idx on public.scans (compactdry_truth);

-- ---------------------------------------------------------------- RLS

alter table public.scans enable row level security;

-- เว็บ (anon key) อ่านได้อย่างเดียว ตามสเปก
drop policy if exists "anon can read scans" on public.scans;
create policy "anon can read scans"
  on public.scans for select
  to anon
  using (true);

-- การเขียนทั้งหมดต้องผ่าน service_role เท่านั้น (ฝั่งเซิร์ฟเวอร์)
-- ไม่ต้องเขียน policy ให้ service_role เพราะมันข้าม RLS อยู่แล้ว
-- การไม่มี policy insert/update/delete สำหรับ anon = anon เขียนไม่ได้

-- ---------------------------------------------------------------- Storage

insert into storage.buckets (id, name, public)
values ('scans', 'scans', false)
on conflict (id) do nothing;

-- bucket เป็น private: เว็บดูภาพผ่าน signed URL ที่เซิร์ฟเวอร์สร้างให้
-- ถ้าอยากให้เปิด public เปลี่ยน public เป็น true แล้วเพิ่ม policy select ให้ anon
drop policy if exists "service role manages scan images" on storage.objects;
create policy "service role manages scan images"
  on storage.objects for all
  to service_role
  using (bucket_id = 'scans')
  with check (bucket_id = 'scans');

-- ---------------------------------------------------------------- keep-alive

-- ปลุก Render ไม่ให้หลับ (free tier หลับหลังไม่มีคนใช้ ~15 นาที) ด้วย pg_cron
-- + pg_net ให้ Supabase ยิง /health ทุก 5 นาที
--
-- ย้ายไปไว้เป็นไฟล์แยกพร้อมรัน: supabase/keepalive.sql (ใส่ URL จริงแล้ว)
-- เปิด SQL Editor วางไฟล์นั้นแล้ว Run ครั้งเดียว
