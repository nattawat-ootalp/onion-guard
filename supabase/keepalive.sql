-- OnionGuard — ปลุก Render ไม่ให้หลับด้วย Supabase (pg_cron + pg_net)
--
-- ปัญหา: Render free tier หลับหลังไม่มี request เข้า ~15 นาที ทำให้ request
-- แรกตอน demo ต้องรอ server ตื่น ~50 วินาที
--
-- วิธีนี้ให้ "ฐานข้อมูล Supabase" (ซึ่งเปิดตลอด) ยิง HTTP ไปที่ /health ของ
-- Render ทุก 5 นาที เป็น request จากภายนอกจริง จึงกันหลับได้ชัวร์กว่าการให้
-- server ยิงหาตัวเอง (self-ping) ที่ Render อาจไม่นับ
--
-- วิธีรัน: Supabase Dashboard > SQL Editor > New query > วางทั้งไฟล์ > Run
-- (รันครั้งเดียวพอ งานจะอยู่ถาวรจนกว่าจะสั่งยกเลิก)

-- 1) เปิด extension ที่จำเป็น (ต้องมีสิทธิ์ owner ซึ่ง SQL Editor มีให้อยู่แล้ว)
create extension if not exists pg_cron;
create extension if not exists pg_net;

-- 2) ตั้งงาน cron ยิง /health ทุก 5 นาที
--    ถ้าเคยตั้งชื่อนี้ไว้แล้วให้ยกเลิกก่อน เพื่อกันงานซ้อน (รันไฟล์นี้ซ้ำได้)
select cron.unschedule('keep-render-awake')
where exists (select 1 from cron.job where jobname = 'keep-render-awake');

select cron.schedule(
  'keep-render-awake',
  '*/5 * * * *',
  $$ select net.http_get(
       url    := 'https://onionguard-api.onrender.com/health',
       timeout_milliseconds := 30000
     ) $$
);

-- ---------------------------------------------------------------- ตรวจสอบ

-- ดูงานที่ตั้งไว้:
--   select jobid, jobname, schedule, active from cron.job;
--
-- ดูประวัติการรันล่าสุด (สำเร็จ/ล้มเหลว):
--   select job_pid, status, return_message, start_time
--   from cron.job_run_details
--   where jobid = (select jobid from cron.job where jobname = 'keep-render-awake')
--   order by start_time desc
--   limit 10;
--
-- ดูผลตอบกลับของ HTTP (pg_net เก็บไว้ชั่วคราว):
--   select id, status_code, content, created
--   from net._http_response
--   order by created desc
--   limit 10;
--
-- ยกเลิกงาน (ถ้าไม่ต้องการแล้ว เช่น ย้ายไป Render แบบเสียเงิน):
--   select cron.unschedule('keep-render-awake');
