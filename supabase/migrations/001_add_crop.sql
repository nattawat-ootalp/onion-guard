-- OnionGuard — เพิ่มคอลัมน์ crop ให้ตาราง scans (รองรับกระเทียม)
--
-- รันไฟล์นี้ใน Supabase Dashboard > SQL Editor > New query > วางแล้ว Run
-- (Supabase ไม่เปิดให้รัน DDL ผ่าน REST API เหมือน schema.sql)
--
-- ทำไมต้องเป็นคอลัมน์ ไม่ใช่ดูจากรหัสตัวอย่าง
-- รหัสตัวอย่างเป็นข้อความที่เจ้าหน้าที่พิมพ์เอง (เคยมี 'hhh', 's002' ตัวพิมพ์เล็ก
-- หลุดเข้ามาแล้ว) ถ้าให้ชนิดพืชขึ้นกับ prefix ของรหัส ชุดข้อมูลเทรนจะเปลี่ยน
-- ไปตามคนพิมพ์ผิด — ซึ่งเป็นความผิดพลาดที่เงียบและตามหายาก
--
-- แถวเดิม 60 แถวเป็นหอมแดงทั้งหมด จึงตั้ง default 'onion' แล้ว backfill ได้เลย
-- ไม่ต้องแก้แถวเก่ามือ

alter table public.scans
  add column if not exists crop text not null default 'onion';

-- จำกัดค่าที่รับได้ ป้องกันคำสะกดต่างกัน ('Garlic', 'garlic ') ทำให้ query
-- ฝั่งเทรนหลุดแถวไปเงียบ ๆ
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'scans_crop_check'
  ) then
    alter table public.scans
      add constraint scans_crop_check check (crop in ('onion', 'garlic'));
  end if;
end $$;

comment on column public.scans.crop is
  'ชนิดพืชของตัวอย่างนี้: onion = หอมแดง, garlic = กระเทียม — โมเดลคนละตัว ห้ามเทรนรวมกันโดยไม่ตั้งใจ';

-- หน้าเว็บและสคริปต์เทรนกรองด้วย crop เกือบทุก query
create index if not exists scans_crop_idx on public.scans (crop);
