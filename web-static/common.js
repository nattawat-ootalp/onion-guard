// Shared chrome (title / mock banner / disclaimer) + a fetch helper used by
// every page in this static site. Loaded after config.js.
(function () {
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // Warm-up ping. Render's free tier sleeps after idle, so the FIRST hit
  // pays a ~50s cold start. Firing /health the instant any page loads lets
  // the host wake in the background while the user is still reading — by the
  // time they press "scan", the server is usually already up. Cheap and
  // best-effort; a failure here changes nothing (the real calls still run).
  function ping() {
    fetch(window.API_BASE + "/health", { cache: "no-store" }).catch(function () {});
  }
  ping();

  // ...แล้วยิงซ้ำทุก 4 นาทีตราบใดที่หน้าเว็บยังเปิดอยู่ และยิงทันทีเมื่อผู้ใช้
  // สลับกลับมาที่แท็บนี้ ระหว่าง demo ที่เปิดหน้าเว็บทิ้งไว้ เซิร์ฟเวอร์จึงไม่มี
  // ช่วงเงียบยาวพอให้ Render สั่งหลับเลย
  setInterval(ping, 4 * 60 * 1000);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) ping();
  });

  fetch(window.API_BASE + "/api/site-info")
    .then(function (r) { return r.json(); })
    .then(function (d) {
      setText("project-title", d.project_title);
      setText("footer-disclaimer", d.disclaimer);
      var banner = document.getElementById("mock-banner");
      if (banner) banner.classList.toggle("hidden", !d.is_mock);
    })
    .catch(function () {
      // Chrome text is cosmetic — a failed fetch here must not block the
      // page's real function (scanning / history), only the header text.
    });

})();

// Render's free tier suspends an idle service, so the FIRST request after a
// while can take up to ~50s to wake it — a plain fetch() would look "stuck"
// with no visible cause. Callers pass a status element to keep whoever is
// waiting informed instead of staring at a frozen button.
function apiFetch(path, options, waitEl) {
  var timer = null;
  if (waitEl) {
    timer = setTimeout(function () {
      waitEl.textContent = "เซิร์ฟเวอร์กำลังตื่นจากโหมดพัก อาจใช้เวลาถึง 1 นาที กรุณารอสักครู่...";
    }, 4000);
  }
  return fetch(window.API_BASE + path, options).finally(function () {
    if (timer) clearTimeout(timer);
  });
}
