/* OnionGuard — โครงร่างร่วม: แถบบน 56px / แถบล่างตรึง 34px ------------ */
(function (root) {
  'use strict';

  var NAV = [
    { href: '/scan', label: 'scan' },
    { href: '/label', label: 'label' },
    { href: '/samples', label: 'samples' },
    { href: '/dataset', label: 'dataset' },
    { href: '/model', label: 'model' },
    { href: '/instrument', label: 'instrument' }
  ];

  var state = {
    queue: 0,
    sync: '14:22',
    node: 'PI-01',
    today: 84
  };

  function here() {
    var p = location.pathname.replace(/\/index\.html$/, '/').replace(/\.html$/, '');
    if (p === '/' || p === '') p = '/scan';
    return p;
  }

  function mount() {
    var cur = here();

    var top = document.createElement('header');
    top.className = 'topbar';
    top.innerHTML =
      '<a class="brand" href="/scan">ONIONGUARD</a>' +
      '<nav class="nav" aria-label="เมนูหลัก">' +
      NAV.map(function (n) {
        var on = n.href === cur ? ' aria-current="page"' : '';
        return '<a href="' + n.href + '"' + on + '>' + n.label + '</a>';
      }).join('') +
      '</nav>' +
      '<span class="dots" title="สถานะกล้องและไฟ UV">' +
      '<span class="dot live"></span><span class="dot warn"></span></span>';

    var foot = document.createElement('footer');
    foot.className = 'statusbar';
    foot.id = 'statusbar';
    document.body.insertBefore(top, document.body.firstChild);
    document.body.appendChild(foot);
    render();
  }

  function render() {
    var f = document.getElementById('statusbar');
    if (!f) return;
    f.innerHTML =
      'คิวรอส่ง <b>' + state.queue + '</b>' +
      '<span class="sep">·</span>ซิงก์ <b>' + state.sync + '</b>' +
      '<span class="sep">·</span><b>' + state.node + '</b>' +
      '<span class="sep">·</span>สแกนวันนี้ <b>' + state.today + '</b> หัว' +
      '<span class="push">v2 · single-angle</span>';
  }

  function bump(key, by) {
    state[key] = (state[key] || 0) + by;
    render();
  }

  /* ---------------------------------------------------------------
     ภาพจากด้านบน: SVG หัวหอมใต้แสง UV + วง ROI เส้นประ + วงรอบจุดผิดปกติ
     สีตระกูล fluor ใช้ได้เฉพาะภายใน svg นี้ (ช่องมองตัวอย่าง)
  ---------------------------------------------------------------- */
  function disc(s, opts) {
    opts = opts || {};
    var size = opts.size || 320;
    var showRoi = opts.roi !== false;
    var showMarks = opts.marks !== false;
    var mode = opts.mode || 'uv';      // uv | dark | white
    var r = root.OG.rng(s.seed || 1);

    // ความเข้มเรืองแสงผูกกับ F_p05 จริง
    var lit = Math.max(0, Math.min(1, (s.fp05 - 0.1) / 0.7));
    var ring = function (t) {
      if (mode === 'dark') return '#15130F';
      if (mode === 'white') return ['#6B5B3E', '#8A7648', '#A08A55', '#B29A61'][t];
      var pal = [
        ['#3A4416', '#4E5C1C', '#5E6E22', '#6E7F28'],
        ['#5C6B1E', '#778A26', '#8EA32C', '#A0B833']
      ];
      var k = lit < 0.45 ? 0 : 1;
      return pal[k][t];
    };

    var c = size / 2, R = size * 0.47;
    var spots = [];
    var n = s.fp05 < 0.36 ? 2 : s.fp05 < 0.52 ? 1 : 0;
    for (var i = 0; i < n; i++) {
      var ang = r() * Math.PI * 2, rad = (0.25 + r() * 0.45) * R;
      spots.push({
        x: c + Math.cos(ang) * rad,
        y: c + Math.sin(ang) * rad,
        rx: (i === 0 ? 0.085 : 0.05) * size * (0.7 + r() * 0.6),
        big: i === 0
      });
    }

    var svg = '<svg viewBox="0 0 ' + size + ' ' + size + '" width="100%" height="100%" ' +
      'preserveAspectRatio="xMidYMid meet" role="img" aria-label="ภาพจากด้านบนของตัวอย่าง ' + s.id +
      ' แสดงวง ROI และจุดผิดปกติ ' + n + ' จุด">' +
      '<rect width="' + size + '" height="' + size + '" fill="#14120F"/>' +
      '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 1.02) + '" fill="#241F18"/>' +
      '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 0.90) + '" fill="' + ring(0) + '"/>' +
      '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 0.76) + '" fill="' + ring(1) + '"/>' +
      '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 0.56) + '" fill="' + ring(2) + '"/>' +
      '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 0.32) + '" fill="' + ring(3) + '"/>';

    spots.forEach(function (p) {
      svg += '<ellipse cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" rx="' + p.rx.toFixed(1) +
        '" ry="' + (p.rx * 0.82).toFixed(1) + '" fill="#413A18"/>' +
        '<ellipse cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" rx="' + (p.rx * 0.55).toFixed(1) +
        '" ry="' + (p.rx * 0.45).toFixed(1) + '" fill="#221E0E"/>';
    });

    if (showRoi) {
      svg += '<circle cx="' + c + '" cy="' + c + '" r="' + (R * 0.90) + '" fill="none" stroke="#EDE7DA" ' +
        'stroke-width="' + (size / 210).toFixed(2) + '" stroke-dasharray="' + (size / 64) + ' ' + (size / 64) + '"/>';
    }
    if (showMarks) {
      spots.forEach(function (p) {
        var col = p.big ? '#A83A12' : '#B5821A';
        svg += '<ellipse cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" rx="' + (p.rx * 1.5).toFixed(1) +
          '" ry="' + (p.rx * 1.32).toFixed(1) + '" fill="none" stroke="' + col + '" stroke-width="' + (size / 150).toFixed(2) + '"/>';
      });
    }
    svg += '</svg>';
    return svg;
  }

  root.Shell = { mount: mount, state: state, render: render, bump: bump, disc: disc };
  document.addEventListener('DOMContentLoaded', mount);
})(window);
