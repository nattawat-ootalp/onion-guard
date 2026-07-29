// Shared chrome (title / mock banner / disclaimer) + a fetch helper used by
// every page in this static site. Loaded after config.js.
(function () {
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

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
