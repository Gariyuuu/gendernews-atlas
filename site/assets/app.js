// GenderNews Atlas — page behaviour: chart mounting, theme toggle, hero figure tick.
import { mountAll } from "./charts.js";

const THEMES = ["system", "light", "dark"];

function readTheme() {
  try { return localStorage.getItem("gna-theme") || "system"; } catch { return "system"; }
}
function applyTheme(t) {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
  const btn = document.querySelector(".theme-toggle");
  if (btn) btn.textContent = `Theme: ${t}`;
}
function initTheme() {
  applyTheme(readTheme());
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;
  btn.hidden = false;
  btn.addEventListener("click", () => {
    const next = THEMES[(THEMES.indexOf(readTheme()) + 1) % THEMES.length];
    try { localStorage.setItem("gna-theme", next); } catch { /* storage unavailable: theme lasts for this page only */ }
    applyTheme(next);
  });
}

// Stat-Led number tick: the server-rendered value is already final; this only animates it.
function tick() {
  const el = document.querySelector("[data-tick]");
  if (!el || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const target = parseFloat(el.dataset.tick);
  const dec = parseInt(el.dataset.decimals || "0", 10);
  if (!Number.isFinite(target)) return;
  const t0 = performance.now(), dur = 500;
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    const e = 1 - (1 - k) ** 4;
    el.firstChild.nodeValue = (target * e).toFixed(dec);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// Interactive filters: a <select data-filter="name"> toggles figures carrying data-filter-name.
function initFilters() {
  for (const bar of document.querySelectorAll("[data-filterbar]")) {
    const selects = [...bar.querySelectorAll("select[data-filter]")];
    const range = bar.querySelector("select[data-range]");
    const scope = document.querySelector(bar.dataset.filterbar);
    const update = () => {
      const want = Object.fromEntries(selects.map((s) => [s.dataset.filter, s.value]));
      let shown = 0;
      for (const fig of scope.querySelectorAll("[data-variant]")) {
        const v = JSON.parse(fig.dataset.variant);
        fig.hidden = !Object.entries(want).every(([k, val]) => v[k] === undefined || v[k] === val);
        shown += fig.hidden ? 0 : 1;
      }
      const empty = scope.querySelector("[data-empty]");
      if (empty) empty.hidden = shown > 0;
      const dom = range ? range.value.split("-").map(Number) : null;
      mountAll(scope, dom ? { x: { domain: [dom[0] - 1.5, dom[1] + 1.5] } } : null);
    };
    if (range) range.addEventListener("change", update);
    selects.forEach((s) => s.addEventListener("change", update));
    update();
  }
}

// The section menu is open in the markup so the nav works without JavaScript;
// on narrow screens, collapse it so 13 links do not push the page below the fold.
function initMenu() {
  const menu = document.querySelector(".mast-menu");
  if (menu && matchMedia("(max-width: 760px)").matches) menu.removeAttribute("open");
}

initMenu();
initTheme();
mountAll();
initFilters();
tick();
