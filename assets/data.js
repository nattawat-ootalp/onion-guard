/* OnionGuard — ข้อมูลจำลอง (deterministic) ------------------------------
   ตัวเลขทั้งหมดในเว็บนี้เป็นข้อมูลตัวอย่างสำหรับ mockup
   ใช้ PRNG แบบมี seed เพื่อให้ทุกหน้าเห็นชุดข้อมูลเดียวกันเสมอ
--------------------------------------------------------------------- */
(function (root) {
  'use strict';

  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
  }

  var TOTAL = 247;
  var TARGET = 250;
  var THRESHOLD = 0.42;          // จุดตัดจาก ROC
  var GROUPS = ['margin', 'core', 'field-A', 'field-B'];
  var TRUTH = ['clear', 'susp', 'pos'];

  var LABEL_TH = { clear: 'ปกติ', susp: 'น่าสงสัย', pos: 'พบเชื้อ', none: 'ยังไม่กรอก' };
  var CLASS_TH = { hardneg: 'Hard negative', neg: 'Negative', pos: 'Positive' };

  /* สร้างรายการตัวอย่าง M-001 .. M-247 */
  function build() {
    var r = rng(20260723);
    var rows = [];
    for (var i = 1; i <= TOTAL; i++) {
      var truthRoll = r();
      var truth = truthRoll < 0.34 ? 'clear' : truthRoll < 0.65 ? 'susp' : 'pos';

      // F_p05: ค่าต่ำ = เรืองแสงหรี่ = มีแนวโน้มพบเชื้อ
      // ค่ากลางของแต่ละคลาสอยู่ใกล้กันและมีการกระจายกว้าง จึงมีบริเวณซ้อนทับจริง
      var base = truth === 'clear' ? 0.59 : truth === 'susp' ? 0.47 : 0.31;
      var fp05 = +(base + (r() - 0.5) * 0.38).toFixed(2);
      fp05 = Math.min(0.92, Math.max(0.08, fp05));

      var aLow = +Math.max(0.1, (0.85 - fp05) * 13 + (r() - 0.5) * 2.4).toFixed(1);
      var blob = Math.max(1, Math.round((0.8 - fp05) * 34 + (r() - 0.5) * 7));

      var pred = fp05 >= THRESHOLD + 0.14 ? 'clear' : fp05 >= THRESHOLD ? 'susp' : 'pos';

      // 62 หัวท้ายคิวยังไม่กรอกผลลอกเปลือก
      var labeled = i <= TOTAL - 62;

      rows.push({
        id: 'M-' + String(i).padStart(3, '0'),
        group: GROUPS[Math.floor(r() * GROUPS.length)],
        time: fmtTime(8 * 60 + Math.round((i / TOTAL) * 6.4 * 60)),
        fp05: fp05,
        aLow: aLow,
        blob: blob,
        pred: pred,
        truth: labeled ? truth : null,
        layers: labeled && truth !== 'clear' ? 1 + Math.floor(r() * 3) : 0,
        cls: null,   // กำหนดภายหลังจาก assignClasses()
        seed: (i * 7919) >>> 0
      });
    }
    return rows;
  }

  function fmtTime(mins) {
    var h = Math.floor(mins / 60) % 24, m = mins % 60;
    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
  }

  var SAMPLES = build();
  var LABELED = SAMPLES.filter(function (s) { return s.truth; });
  var PENDING = SAMPLES.filter(function (s) { return !s.truth; });

  /* กำหนดคลาสสำหรับแดชบอร์ดชุดข้อมูล ให้สัดส่วนสมจริง
     pos = หัวที่ผ่าแล้วพบเชื้อ
     hard negative = หัวที่ไม่พบเชื้อแต่เรืองแสงหรี่ (F_p05 ต่ำ) จึงแยกยาก
     negative = หัวปกติที่เรืองแสงชัดเจน                                */
  (function assignClasses() {
    LABELED.forEach(function (s) { s.cls = s.truth === 'pos' ? 'pos' : 'neg'; });
    var nonPos = LABELED.filter(function (s) { return s.truth !== 'pos'; })
      .sort(function (a, b) { return a.fp05 - b.fp05; });
    var hard = Math.round(nonPos.length * 0.46);   // ~46% ของหัวที่ไม่พบเชื้อคือ hard negative
    for (var i = 0; i < hard; i++) nonPos[i].cls = 'hardneg';
    PENDING.forEach(function (s) {
      s.cls = s.pred === 'pos' ? 'pos' : (s.fp05 < 0.5 ? 'hardneg' : 'neg');
    });
  })();

  function counts() {
    var c = { hardneg: 0, neg: 0, pos: 0 };
    LABELED.forEach(function (s) { c[s.cls]++; });
    return c;
  }

  /* ฮิสโตแกรมของ F_p05 แยกตามคลาส (16 ช่อง จาก 0 ถึง 1) */
  function histogram() {
    var bins = 16, a = new Array(bins).fill(0), b = new Array(bins).fill(0);
    LABELED.forEach(function (s) {
      var k = Math.min(bins - 1, Math.floor(s.fp05 * bins));
      if (s.truth === 'pos') b[k]++; else a[k]++;
    });
    return { bins: bins, clear: a, pos: b };
  }

  /* ROC จากค่า F_p05 (กลับทิศ: ค่าน้อย = positive) */
  function roc() {
    var pts = [], scored = LABELED.map(function (s) {
      return { score: 1 - s.fp05, y: s.truth === 'pos' ? 1 : 0 };
    }).sort(function (x, y) { return y.score - x.score; });

    var P = scored.filter(function (s) { return s.y; }).length;
    var N = scored.length - P, tp = 0, fp = 0;
    pts.push({ x: 0, y: 0, cut: 1 });
    scored.forEach(function (s) {
      if (s.y) tp++; else fp++;
      pts.push({ x: fp / N, y: tp / P, cut: +(1 - s.score).toFixed(3) });
    });

    var auc = 0;
    for (var i = 1; i < pts.length; i++) {
      auc += (pts[i].x - pts[i - 1].x) * (pts[i].y + pts[i - 1].y) / 2;
    }
    return { pts: pts, auc: auc };
  }

  /* confusion matrix ที่จุดตัด cut (ทำนาย positive เมื่อ fp05 < cut) */
  function confusion(cut) {
    var tp = 0, fp = 0, tn = 0, fn = 0;
    LABELED.forEach(function (s) {
      var p = s.fp05 < cut, y = s.truth === 'pos';
      if (p && y) tp++; else if (p && !y) fp++; else if (!p && y) fn++; else tn++;
    });
    return {
      tp: tp, fp: fp, tn: tn, fn: fn,
      sens: tp + fn ? tp / (tp + fn) : 0,
      spec: tn + fp ? tn / (tn + fp) : 0
    };
  }

  root.OG = {
    TOTAL: TOTAL, TARGET: TARGET, THRESHOLD: THRESHOLD,
    GROUPS: GROUPS, TRUTH: TRUTH, LABEL_TH: LABEL_TH, CLASS_TH: CLASS_TH,
    samples: SAMPLES, labeled: LABELED, pending: PENDING,
    counts: counts, histogram: histogram, roc: roc, confusion: confusion,
    rng: rng, fmtTime: fmtTime
  };
})(window);
