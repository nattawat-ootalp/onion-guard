-- OnionGuard — เพิ่มคอลัมน์ crop ให้ตาราง public_scans (เว็บสาธารณะสแกนกระเทียมได้)
--
-- รันไฟล์นี้ใน Supabase Dashboard > SQL Editor > New query > วางแล้ว Run
-- (Supabase ไม่เปิดให้รัน DDL ผ่าน REST API เหมือน schema.sql และ 001)
--
-- ไม่ใช่ migration ที่ต้องรีบ: เว็บสาธารณะสแกนกระเทียมได้อยู่แล้วโดยไม่ต้องรันไฟล์นี้
--
-- ชนิดพืชคือสิ่งเดียวในแถวที่ย้อนกลับไปคำนวณใหม่ไม่ได้ จึงต้องถูกบันทึกเสมอ
-- คำถามมีแค่ว่า "เก็บตรงไหน" — ถ้ายังไม่รันไฟล์นี้ เซิร์ฟเวอร์จะเก็บไว้ใน
-- quality_notes (jsonb ที่มีอยู่แล้ว) และหน้าประวัติอ่านจากตรงนั้นให้เอง
-- ดู _save_scan / api_public_scans ใน web/app.py
--
-- สิ่งที่ได้เพิ่มจากการรันไฟล์นี้คือคุณสมบัติของคอลัมน์จริง: query/index ได้
-- ตรง ๆ และมี check constraint กันคำสะกดเพี้ยน ('Garlic', 'garlic ') ซึ่ง jsonb
-- ให้ไม่ได้ เซิร์ฟเวอร์ตรวจเองว่ามีคอลัมน์หรือยังแล้วสลับที่เก็บให้อัตโนมัติ
-- (แคชผลตรวจไว้ ~2 นาที) จึงรันตอนไหนก็ได้ ไม่ต้องหยุดเว็บ ไม่ต้อง deploy ใหม่

alter table public.public_scans
  add column if not exists crop text not null default 'onion';

-- จำกัดค่าที่รับได้เหมือนตาราง scans ('Garlic', 'garlic ' ที่สะกดต่างกัน
-- จะทำให้ query ฝั่งหน้าประวัติหลุดแถวไปเงียบ ๆ)
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'public_scans_crop_check'
  ) then
    alter table public.public_scans
      add constraint public_scans_crop_check check (crop in ('onion', 'garlic'));
  end if;
end $$;

comment on column public.public_scans.crop is
  'ชนิดพืชของสแกนนี้: onion = หอมแดง, garlic = กระเทียม — คัดกรองด้วยโมเดลคนละตัว';

create index if not exists public_scans_crop_idx on public.public_scans (crop);
