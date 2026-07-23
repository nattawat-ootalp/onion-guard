# OnionGuard — UX/UI Mockup v2 (single-angle)

เว็บ mockup ตาม UX/UI Specification v2 ของ OnionGuard — ระบบสแกนหัวหอมด้วยแสง UV
โดยถ่าย **มุมเดียวจากด้านบน** แล้วแปลงความเข้มเรืองแสงของแต่ละพิกเซลเป็น **ภูมิประเทศ**
เพื่อให้จุดติดเชื้อกลายเป็นหลุมที่มองเห็นได้ทันที

เว็บนี้เป็น mockup — ตัวเลขทั้งหมดเป็นข้อมูลจำลองแบบ deterministic (`assets/data.js`)

## หน้าจอ (6 หน้า)

| เส้นทาง | ชื่อหน้า | หน้าที่ |
|---|---|---|
| `/scan` | แผงสแกน | วางหัวหอม กด ดูผล — ลดจำนวนการแตะต่อหัวให้น้อยที่สุด |
| `/label` | คิวกรอกผลลอกเปลือก | บันทึกค่าความจริงหลังผ่า สองแตะจบหนึ่งรายการ |
| `/samples` | รายการตัวอย่าง | ตารางทุกหัวที่สแกน ค้นหา กรอง ส่งออก CSV |
| `/dataset` | แดชบอร์ดชุดข้อมูล | คุมสัดส่วนคลาส + ฮิสโตแกรม F_p05 |
| `/model` | ผลการประเมิน | ROC เลื่อนจุดตัดได้ + confusion matrix |
| `/instrument` | ห้องเครื่อง | กล่องถ่ายภาพสามมิติผ่าครึ่ง |

## หลักการออกแบบ (ข้อห้าม)

- ไม่มี gradient / ไม่มีสีม่วง–น้ำเงินอมม่วง / รัศมีมุมไม่เกิน 3px
- ไม่มีวงกลมโหลดหมุน — ใช้ข้อความบอกขั้นตอนแทน
- สี fluor ใช้ได้เฉพาะในช่องมองตัวอย่าง (`.stage`) เท่านั้น
- ภูมิประเทศ 3D ต้องแสดงค่าจริงคู่กับตัวคูณความสูงเสมอ

## รันในเครื่อง

```bash
npx serve -l 3000 .
# เปิด http://localhost:3000/scan
```

## Deploy

Static site บน Vercel — `vercel.json` ตั้ง `cleanUrls` ให้ `/scan.html` เสิร์ฟที่ `/scan`

## โครงสร้าง

```
index.html          redirect -> /scan
scan.html label.html samples.html dataset.html model.html instrument.html
assets/
  style.css   ระบบดีไซน์ (โทเคนสี, คอมโพเนนต์)
  data.js     ข้อมูลจำลอง deterministic + สถิติ (ROC, histogram, confusion)
  shell.js    แถบบน/ล่างร่วม + Shell.disc() ภาพจากด้านบน (SVG)
  relief.js   ภูมิประเทศเรืองแสง 3D (Three.js)
```
