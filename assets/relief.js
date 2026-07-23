/* OnionGuard — ภูมิประเทศเรืองแสง (3D relief) --------------------------
   ความสูง = ความเข้มเรืองแสงของแต่ละพิกเซล
   ระนาบสีสนิม = จุดตัดจาก ROC · ทุกอย่างที่จมใต้ระนาบถูกนับเป็นผิดปกติ
   หมายเหตุ: ความสูงถูกขยาย HEIGHT_X เท่า ต้องแสดงตัวคูณกำกับเสมอ
--------------------------------------------------------------------- */
(function (root) {
  'use strict';

  var HEIGHT_X = 3.0;

  function toonRamp(n) {
    var d = new Uint8Array(n * 3);
    for (var i = 0; i < n; i++) {
      var v = Math.round(58 + (i / (n - 1)) * 197);
      d[i * 3] = v; d[i * 3 + 1] = v; d[i * 3 + 2] = v;
    }
    var t = new THREE.DataTexture(d, n, 1, THREE.RGBFormat);
    t.minFilter = THREE.NearestFilter;
    t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  }

  function create(opts) {
    var canvas = opts.canvas;
    if (!root.THREE || !canvas) return null;

    var s = opts.sample;
    var reduce = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;

    var W = canvas.clientWidth || 800, H = canvas.clientHeight || 400;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
    renderer.setSize(W, H, false);
    renderer.setClearColor(0x14120F, 1);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(32, W / H, 0.1, 100);
    var group = new THREE.Group();
    scene.add(group);

    var R = 1.85, SEG = 88;
    var geo = new THREE.PlaneGeometry(4.2, 4.2, SEG, SEG);

    /* ระดับพื้นฐานมาจาก F_p05 จริงของตัวอย่าง */
    var plateau = 0.30 + (s.fp05 - 0.1) * 0.72;
    var pits = [];
    var r = root.OG.rng(s.seed || 1);
    var nPits = s.fp05 < 0.36 ? 2 : s.fp05 < 0.52 ? 1 : 0;
    for (var k = 0; k < nPits; k++) {
      var ang = r() * Math.PI * 2, rad = (0.25 + r() * 0.45) * R;
      pits.push({
        x: Math.cos(ang) * rad,
        y: Math.sin(ang) * rad,
        depth: (k === 0 ? 0.60 : 0.26) * (0.7 + r() * 0.6),
        width: k === 0 ? 0.075 : 0.030
      });
    }

    function field(x, y) {
      var rr = Math.sqrt(x * x + y * y);
      if (rr > R) return -0.34;
      var h = plateau;
      h += 0.035 * Math.sin(x * 3.1) * Math.cos(y * 2.7);   // ผิวไม่เรียบสนิท
      h += 0.022 * Math.sin(x * 6.4 + 1.2) * Math.sin(y * 5.9);
      for (var i = 0; i < pits.length; i++) {
        var p = pits[i], dx = x - p.x, dy = y - p.y;
        h -= p.depth * Math.exp(-(dx * dx + dy * dy) / p.width);
      }
      var edge = Math.min(1, (R - rr) / 0.30);              // ขอบ ROI ลาดลง
      return h * edge + (-0.34) * (1 - edge);
    }

    var pos = geo.attributes.position;
    for (var i = 0; i < pos.count; i++) {
      pos.setZ(i, field(pos.getX(i), pos.getY(i)));
    }
    geo.computeVertexNormals();

    var terrain = new THREE.Mesh(geo, new THREE.MeshToonMaterial({
      color: 0xB9CC3C, gradientMap: toonRamp(5)
    }));
    terrain.rotation.x = -Math.PI / 2;
    group.add(terrain);

    /* วง ROI */
    var ring = new THREE.Mesh(
      new THREE.RingGeometry(R - 0.015, R + 0.015, 96),
      new THREE.MeshBasicMaterial({ color: 0xEDE7DA, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = plateau + 0.04;
    group.add(ring);

    /* ระนาบเกณฑ์ */
    var plane = new THREE.Mesh(
      new THREE.PlaneGeometry(4.2, 4.2),
      new THREE.MeshBasicMaterial({
        color: 0xA83A12, transparent: true, opacity: 0.42, side: THREE.DoubleSide
      })
    );
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = 0.30 + (root.OG.THRESHOLD - 0.1) * 0.72;
    plane.visible = opts.plane !== false;
    group.add(plane);

    var key = new THREE.DirectionalLight(0xffffff, 1.05);
    key.position.set(2.6, 4.2, 2.2);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0xffffff, 0.40));

    var az = -0.62, el = 0.72, dist = 6.4, tAz = az, tEl = el;
    function place() {
      camera.position.set(
        Math.sin(az) * Math.cos(el) * dist,
        Math.sin(el) * dist,
        Math.cos(az) * Math.cos(el) * dist
      );
      camera.lookAt(0, 0.15, 0);
    }

    var drag = false, px = 0, py = 0;
    function down(x, y) { drag = true; px = x; py = y; }
    function move(x, y) {
      if (!drag) return;
      tAz -= (x - px) * 0.008;
      tEl = Math.max(0.18, Math.min(1.35, tEl + (y - py) * 0.006));
      px = x; py = y;
    }
    function up() { drag = false; }

    canvas.addEventListener('mousedown', function (e) { down(e.clientX, e.clientY); });
    root.addEventListener('mousemove', function (e) { move(e.clientX, e.clientY); });
    root.addEventListener('mouseup', up);
    canvas.addEventListener('touchstart', function (e) { var t = e.touches[0]; down(t.clientX, t.clientY); }, { passive: true });
    canvas.addEventListener('touchmove', function (e) { var t = e.touches[0]; move(t.clientX, t.clientY); e.preventDefault(); }, { passive: false });
    canvas.addEventListener('touchend', up);

    function resize() {
      var w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    root.addEventListener('resize', resize);

    var api = {
      active: true,
      setPlane: function (on) { plane.visible = on; },
      resize: resize,
      dispose: function () { api.active = false; renderer.dispose(); }
    };

    var visible = true;
    if ('IntersectionObserver' in root) {
      new IntersectionObserver(function (es) { visible = es[0].isIntersecting; }, { threshold: 0.03 }).observe(canvas);
    }

    function loop() {
      if (!api.active) return;
      requestAnimationFrame(loop);
      if (!visible || !api.enabled) return;
      az += (tAz - az) * (reduce ? 1 : 0.12);
      el += (tEl - el) * (reduce ? 1 : 0.12);
      place();
      renderer.render(scene, camera);
    }

    api.enabled = opts.enabled !== false;
    resize(); place(); loop();
    return api;
  }

  root.OGRelief = { create: create, HEIGHT_X: HEIGHT_X };
})(window);
