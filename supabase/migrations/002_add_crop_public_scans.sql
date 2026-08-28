-- OnionGuard — เพิ่มคอลัมน์ crop ให้ตาราง public_scans (เว็บสาธารณะสแกนกระเทียมได้)
--
-- รันไฟล์นี้ใน Supabase Dashboard > SQL Editor > New query > วางแล้ว Run
-- (Supabase ไม่เปิดให้รัน DDL ผ่าน REST API เหมือน schema.sql และ 001)
--
-- ทำไมต้องรันก่อนถึงจะเลือกกระเทียมในเว็บสาธารณะได้
-- ถ้าไม่มีคอลัมน์นี้ แถวของกระเทียมกับหอมแดงจะแยกกันไม่ออกเลย — ผลคัดกรอง
-- ของสองพืชมาจากโมเดลคนละตัว (หอมแดงเป็นตัวจำแนกจากผลเพาะเชื้อ กระเทียมเป็น
-- การเทียบกับกลีบปกติ) ถ้าเก็บปนกันโดยไม่รู้ว่าแถวไหนเป็นอะไร หน้าประวัติ
-- จะรายงานผิด และย้อนกลับไปแก้ทีหลังไม่ได้ เพราะข้อมูลที่หายคือ "ชนิดพืช"
-- ไม่ใช่ค่าที่คำนวณใหม่ได้
--
-- เซิร์ฟเวอร์ตรวจเองว่าคอลัมน์นี้มีหรือยัง ถ้ายังไม่มีจะไม่เปิดให้เลือกกระเทียม
-- ในเว็บสาธารณะ (เว็บยังทำงานปกติ แค่มีหอมแดงอย่างเดียวเหมือนเดิม) จึงรัน
-- migration นี้ตอนไหนก็ได้ ไม่ต้องหยุดเว็บ

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
