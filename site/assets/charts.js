// GenderNews Atlas — dependency-free SVG chart kit.
//
// Every figure on the site is rendered by this module from a spec that
// site/build.py embeds next to the figure (<script type="application/json">).
// The build step also writes a table view for each figure, so no value is
// reachable only through a tooltip.
//
// Mark specs (dataviz method + portfolio MASTER.css):
//   2px round lines · >= 8px end dots with a 2px surface ring · CI as a 12% wash
//   solid hairline grid · every series also carries a dash pattern
//   text in ink tokens, never in series colour · legend whenever >= 2 series
//   colours resolve through CSS custom properties, so theme switches need no re-render

const NS = "http://www.w3.org/2000/svg";
export const COLOR = {
  women: "var(--viz-women)", men: "var(--viz-men)", neutral: "var(--viz-neutral)",
  accent: "var(--color-accent)",
};
export const DASH = { 1: "none", 2: "6 4", 3: "1 4", 4: "10 4 2 4", 5: "2 2" };
// Sequential ramp (single hue, light -> dark); validated reference ramp.
export const SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

const MINUS = "−";
export const fmt = {
  pct: (v) => (v == null || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(1)}%`),
  pct0: (v) => (v == null || Number.isNaN(v) ? "—" : `${Math.round(100 * v)}%`),
  pp: (v) => (v == null || Number.isNaN(v) ? "—" : `${v > 0 ? "+" : v < 0 ? MINUS : ""}${Math.abs(v).toFixed(2)} pp`),
  num: (v) => (v == null || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString("en-US")),
  dec2: (v) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(2).replace("-", MINUS)),
  year: (v) => String(Math.round(v)),
};

// ---------------------------------------------------------------- pure helpers (unit-tested)
export function linear(d0, d1, r0, r1) {
  const span = d1 - d0 || 1;
  const k = (r1 - r0) / span;
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => d0 + (p - r0) / k;
  return f;
}

export function niceTicks(min, max, n = 5) {
  if (!(max > min)) return [min];
  const raw = (max - min) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const err = raw / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

export function extent(values, pad = 0) {
  const v = values.filter((x) => x != null && Number.isFinite(x));
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

export function nearestIndex(xs, x) {
  let best = 0, bd = Infinity;
  xs.forEach((v, i) => { const d = Math.abs(v - x); if (d < bd) { bd = d; best = i; } });
  return best;
}

export function seqColor(t) {
  const c = Math.max(0, Math.min(1, t));
  return SEQ[Math.round(c * (SEQ.length - 1))];
}

export function inkOn(hex) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  const L = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  return L > 0.36 ? "var(--viz-cell-ink-dark)" : "var(--viz-cell-ink-light)";
}

// ---------------------------------------------------------------- DOM helpers
function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function htmlEl(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function pathD(pts, x, y) {
  let d = "", pen = false;
  for (const p of pts) {
    if (p.y == null || !Number.isFinite(p.y)) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`;
    pen = true;
  }
  return d;
}
function bandD(pts, x, y) {
  const ok = pts.filter((p) => p.lo != null && p.hi != null && Number.isFinite(p.lo) && Number.isFinite(p.hi));
  if (ok.length < 2) return "";
  const top = ok.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.hi).toFixed(1)}`).join("");
  const bot = ok.slice().reverse().map((p) => `L${x(p.x).toFixed(1)},${y(p.lo).toFixed(1)}`).join("");
  return `${top}${bot}Z`;
}
function legend(host, series) {
  if (series.length < 2) return;
  const lg = htmlEl("ul", "viz-legend", host);
  for (const s of series) {
    const li = htmlEl("li", null, lg);
    const k = svgEl("svg", { width: 22, height: 10, "aria-hidden": "true" }, li);
    svgEl("line", { x1: 1, y1: 5, x2: 21, y2: 5, style: `stroke:${COLOR[s.color] || s.color};stroke-width:2;stroke-dasharray:${DASH[s.dash || 1]};stroke-linecap:round` }, k);
    htmlEl("span", null, li, s.label);
  }
}
function tooltip(host) {
  const t = htmlEl("div", "viz-tip", host);
  t.hidden = true;
  t.setAttribute("role", "status");
  return t;
}
function axisY(g, y, ticks, W, left, format) {
  for (const t of ticks) {
    svgEl("line", { x1: left, x2: W, y1: y(t), y2: y(t), class: "viz-grid" }, g);
    const lab = svgEl("text", { x: left - 6, y: y(t), class: "viz-tick", "text-anchor": "end", "dominant-baseline": "middle" }, g);
    lab.textContent = format(t);
  }
}
function axisX(g, x, ticks, y0, format) {
  for (const t of ticks) {
    const lab = svgEl("text", { x: x(t), y: y0 + 16, class: "viz-tick", "text-anchor": "middle" }, g);
    lab.textContent = format(t);
  }
  svgEl("line", { x1: x(ticks[0]) - 4, x2: x(ticks[ticks.length - 1]) + 4, y1: y0, y2: y0, class: "viz-axis" }, g);
}

// ---------------------------------------------------------------- line chart
export function lineChart(host, spec, opts = {}) {
  host.replaceChildren();
  const series = spec.series.filter((s) => s.points && s.points.length);
  if (!series.length) { htmlEl("p", "viz-empty", host, "No data for this view."); return; }
  if (!opts.compact) legend(host, series);
  const W = Math.max(260, host.clientWidth || 640);
  const narrow = W < 520;
  const H = opts.height || spec.height || (opts.compact ? 150 : narrow ? 240 : 300);
  const endLabels = !opts.compact && !narrow && series.length <= 4 && spec.endLabels !== false;
  const M = { t: 10, r: endLabels ? 132 : 14, b: 26, l: opts.compact ? 34 : 44 };
  const fmtY = fmt[spec.y?.format || "pct0"];
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const xDom = spec.x?.domain || extent(xs, 0.02);
  const yVals = series.flatMap((s) => s.points.flatMap((p) => [p.y, p.lo, p.hi]));
  const yDom = spec.y?.domain || extent(yVals, 0.08);
  const yTicks = niceTicks(yDom[0], yDom[1], opts.compact ? 3 : 5);
  const yLo = Math.min(yDom[0], yTicks[0]), yHi = Math.max(yDom[1], yTicks[yTicks.length - 1]);
  const x = linear(xDom[0], xDom[1], M.l, W - M.r);
  const y = linear(yLo, yHi, H - M.b, M.t);
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz-svg", role: "img",
    "aria-label": `${spec.title || "Chart"}. Values are listed in the table view.`, tabindex: opts.compact ? null : "0" }, host);
  const g = svgEl("g", {}, svg);
  axisY(g, y, yTicks, W - M.r, M.l, fmtY);
  const xTicks = spec.x?.ticks || niceTicks(xDom[0], xDom[1], opts.compact ? 3 : 6).filter((t) => t >= xDom[0] && t <= xDom[1]);
  axisX(g, x, xTicks, H - M.b, fmt.year);
  for (const r of spec.refLines || []) {
    svgEl("line", { x1: x(r.x), x2: x(r.x), y1: M.t, y2: H - M.b, class: "viz-ref" }, g);
    if (!opts.compact) {
      const t = svgEl("text", { x: x(r.x) + 4, y: M.t + 10, class: "viz-note" }, g);
      t.textContent = r.label;
    }
  }
  if (spec.zero) svgEl("line", { x1: M.l, x2: W - M.r, y1: y(0), y2: y(0), class: "viz-ref" }, g);
  const labels = [];
  for (const s of series) {
    const col = COLOR[s.color] || s.color;
    const band = bandD(s.points, x, y);
    if (band) svgEl("path", { d: band, style: `fill:${col};opacity:.12` }, g);
    svgEl("path", { d: pathD(s.points, x, y), class: "viz-line",
      style: `stroke:${col};stroke-width:${s.width || 2};stroke-dasharray:${DASH[s.dash || 1]}` }, g);
    const last = [...s.points].reverse().find((p) => p.y != null && Number.isFinite(p.y));
    if (last) {
      svgEl("circle", { cx: x(last.x), cy: y(last.y), r: opts.compact ? 3.5 : 4.5, class: "viz-dot", style: `fill:${col}` }, g);
      labels.push({ s, yy: y(last.y), xx: x(last.x) });
    }
  }
  if (endLabels) {
    labels.sort((a, b) => a.yy - b.yy);
    const collide = labels.some((l, i) => i && l.yy - labels[i - 1].yy < 13);
    if (!collide) for (const l of labels) {
      const t = svgEl("text", { x: l.xx + 9, y: l.yy, class: "viz-endlabel", "dominant-baseline": "middle" }, g);
      t.textContent = l.s.label;
    }
  }
  if (opts.compact) return;

  // crosshair + tooltip (pointer and keyboard)
  const tip = tooltip(host);
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  const cross = svgEl("line", { y1: M.t, y2: H - M.b, class: "viz-cross", visibility: "hidden" }, g);
  const marks = series.map((s) => svgEl("circle", { r: 5, class: "viz-dot", visibility: "hidden", style: `fill:${COLOR[s.color] || s.color}` }, g));
  let idx = allX.length - 1;
  const show = (i) => {
    idx = Math.max(0, Math.min(allX.length - 1, i));
    const xv = allX[idx];
    cross.setAttribute("x1", x(xv)); cross.setAttribute("x2", x(xv)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren();
    htmlEl("strong", null, tip, fmt.year(xv));
    series.forEach((s, k) => {
      const p = s.points.find((q) => q.x === xv);
      if (!p || p.y == null) { marks[k].setAttribute("visibility", "hidden"); return; }
      marks[k].setAttribute("cx", x(xv)); marks[k].setAttribute("cy", y(p.y)); marks[k].setAttribute("visibility", "visible");
      const row = htmlEl("span", "viz-tip-row", tip);
      const key = htmlEl("i", "viz-key", row);
      key.style.background = COLOR[s.color] || s.color;
      const ci = p.lo != null && p.hi != null ? ` (${fmtY(p.lo)}–${fmtY(p.hi)})` : "";
      const n = p.n != null ? ` · n=${fmt.num(p.n)}` : "";
      row.append(`${s.label}: ${fmtY(p.y)}${ci}${n}`);
    });
    tip.hidden = false;
    const px = x(xv);
    tip.style.left = `${Math.min(Math.max(px + 12, 0), W - 230)}px`;
    tip.style.top = `${M.t + 4}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); marks.forEach((m) => m.setAttribute("visibility", "hidden")); };
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    show(nearestIndex(allX, x.invert((e.clientX - r.left) * (W / r.width))));
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(idx));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
    else if (e.key === "Escape") hide();
  });
}

// ---------------------------------------------------------------- small multiples
export function smallMultiples(host, spec) {
  host.replaceChildren();
  const first = spec.panels.find((p) => p.series.length);
  if (first) legend(host, first.series);
  const grid = htmlEl("div", "viz-multiples", host);
  const yVals = spec.panels.flatMap((p) => p.series.flatMap((s) => s.points.flatMap((q) => [q.y, q.lo, q.hi])));
  const yDom = spec.y?.domain || extent(yVals, 0.05);
  for (const p of spec.panels) {
    const cell = htmlEl("div", "viz-panel", grid);
    htmlEl("h4", "viz-panel-title", cell, p.title);
    const plot = htmlEl("div", "viz-panel-plot", cell);
    lineChart(plot, { ...spec, series: p.series, y: { ...(spec.y || {}), domain: yDom }, title: p.title }, { compact: true });
    if (p.note) htmlEl("p", "viz-panel-note", cell, p.note);
  }
}

// ---------------------------------------------------------------- dot plot (estimates with CI)
export function dotPlot(host, spec) {
  host.replaceChildren();
  const rows = spec.rows;
  if (spec.legend) legend(host, spec.legend);
  const W = Math.max(280, host.clientWidth || 640);
  const labelW = Math.min(spec.labelWidth || 190, W * 0.42);
  const rowH = 30, M = { t: 8, r: 70, b: 30, l: labelW + 10 };
  const H = M.t + rows.length * rowH + M.b;
  const vals = rows.flatMap((r) => [r.value, r.lo, r.hi]);
  const dom = extent(spec.zero ? [...vals, 0] : vals, 0.08);
  const ticks = niceTicks(dom[0], dom[1], 5);
  const x = linear(Math.min(dom[0], ticks[0]), Math.max(dom[1], ticks[ticks.length - 1]), M.l, W - M.r);
  const f = fmt[spec.format || "dec2"];
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz-svg", role: "img",
    "aria-label": `${spec.title || "Estimates"}. Values are listed in the table view.` }, host);
  const g = svgEl("g", {}, svg);
  for (const t of ticks) {
    svgEl("line", { x1: x(t), x2: x(t), y1: M.t, y2: H - M.b, class: "viz-grid" }, g);
    const lab = svgEl("text", { x: x(t), y: H - M.b + 16, class: "viz-tick", "text-anchor": "middle" }, g);
    lab.textContent = f(t);
  }
  if (spec.zero) svgEl("line", { x1: x(0), x2: x(0), y1: M.t, y2: H - M.b, class: "viz-ref" }, g);
  if (spec.xLabel) {
    const t = svgEl("text", { x: W - M.r, y: H - 2, class: "viz-note", "text-anchor": "end" }, g);
    t.textContent = spec.xLabel;
  }
  const tip = tooltip(host);
  rows.forEach((r, i) => {
    const cy = M.t + i * rowH + rowH / 2;
    const col = COLOR[r.color || spec.color || "women"];
    const lab = svgEl("text", { x: M.l - 10, y: cy, class: "viz-rowlabel", "text-anchor": "end", "dominant-baseline": "middle" }, g);
    lab.textContent = r.label;
    if (r.lo != null && r.hi != null && Number.isFinite(r.lo) && Number.isFinite(r.hi))
      svgEl("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, class: "viz-whisker", style: `stroke:${col}` }, g);
    if (r.value != null && Number.isFinite(r.value)) {
      svgEl("circle", { cx: x(r.value), cy, r: 5, class: "viz-dot",
        style: r.hollow ? `fill:var(--color-paper);stroke:${col};stroke-width:2` : `fill:${col}` }, g);
      const v = svgEl("text", { x: W - M.r + 8, y: cy, class: "viz-value", "dominant-baseline": "middle" }, g);
      v.textContent = f(r.value);
    }
    const hit = svgEl("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, class: "viz-hit", tabindex: "0" }, g);
    const on = () => {
      tip.replaceChildren();
      htmlEl("strong", null, tip, r.label);
      htmlEl("span", "viz-tip-row", tip, `${f(r.value)}${r.lo != null ? ` (95% CI ${f(r.lo)} to ${f(r.hi)})` : ""}${r.note ? ` · ${r.note}` : ""}`);
      tip.hidden = false;
      tip.style.left = `${Math.min(x(r.value ?? 0) + 12, W - 240)}px`;
      tip.style.top = `${cy + 10}px`;
    };
    hit.addEventListener("pointerenter", on);
    hit.addEventListener("focus", on);
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
    hit.addEventListener("blur", () => { tip.hidden = true; });
  });
}

// ---------------------------------------------------------------- heatmap (printed values)
export function heatmap(host, spec) {
  host.replaceChildren();
  const { rows, cols, cells } = spec;
  const W = Math.max(280, host.clientWidth || 640);
  const labelW = Math.min(spec.labelWidth || 200, W * 0.45);
  const cw = Math.max(38, (W - labelW - 8) / cols.length), ch = 26;
  const H = 28 + rows.length * ch;
  const f = fmt[spec.format || "dec2"];
  const vmin = spec.vmin ?? 0, vmax = spec.vmax ?? 1;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz-svg", role: "img",
    "aria-label": `${spec.title || "Heatmap"}. Values are printed in each cell and listed in the table view.` }, host);
  cols.forEach((c, j) => {
    const t = svgEl("text", { x: labelW + j * cw + cw / 2, y: 16, class: "viz-tick", "text-anchor": "middle" }, svg);
    t.textContent = c;
  });
  const tip = tooltip(host);
  rows.forEach((r, i) => {
    const yy = 24 + i * ch;
    const t = svgEl("text", { x: labelW - 8, y: yy + ch / 2, class: "viz-rowlabel", "text-anchor": "end", "dominant-baseline": "middle" }, svg);
    t.textContent = r;
    cols.forEach((c, j) => {
      const v = cells[i][j];
      const xx = labelW + j * cw;
      if (v == null || !Number.isFinite(v)) {
        const e = svgEl("text", { x: xx + cw / 2, y: yy + ch / 2, class: "viz-tick", "text-anchor": "middle", "dominant-baseline": "middle" }, svg);
        e.textContent = "—";
        return;
      }
      const fill = seqColor((v - vmin) / (vmax - vmin || 1));
      const rect = svgEl("rect", { x: xx + 1, y: yy + 1, width: cw - 2, height: ch - 2, rx: 2, style: `fill:${fill}`, tabindex: "0", class: "viz-cell" }, svg);
      const lab = svgEl("text", { x: xx + cw / 2, y: yy + ch / 2, class: "viz-celltext", "text-anchor": "middle", "dominant-baseline": "middle", style: `fill:${inkOn(fill)}` }, svg);
      lab.textContent = f(v);
      const on = () => {
        tip.replaceChildren();
        htmlEl("strong", null, tip, `${r} · ${c}`);
        htmlEl("span", "viz-tip-row", tip, `${spec.valueLabel || "value"}: ${f(v)}${spec.counts ? ` · n=${fmt.num(spec.counts[i][j])}` : ""}`);
        tip.hidden = false;
        tip.style.left = `${Math.min(xx + cw, W - 220)}px`;
        tip.style.top = `${yy + ch}px`;
      };
      rect.addEventListener("pointerenter", on);
      rect.addEventListener("focus", on);
      rect.addEventListener("pointerleave", () => { tip.hidden = true; });
      rect.addEventListener("blur", () => { tip.hidden = true; });
    });
  });
}

// ---------------------------------------------------------------- horizontal bars (signed, with CI)
export function bars(host, spec) {
  host.replaceChildren();
  const rows = spec.rows;
  const W = Math.max(280, host.clientWidth || 640);
  const labelW = Math.min(spec.labelWidth || 220, W * 0.45);
  const rowH = 28, thick = 14, M = { t: 6, r: 64, b: 28, l: labelW + 10 };
  const H = M.t + rows.length * rowH + M.b;
  const dom = extent([0, ...rows.flatMap((r) => [r.value, r.lo, r.hi])], 0.08);
  const ticks = niceTicks(dom[0], dom[1], 5);
  const x = linear(Math.min(dom[0], ticks[0]), Math.max(dom[1], ticks[ticks.length - 1]), M.l, W - M.r);
  const f = fmt[spec.format || "pp"];
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz-svg", role: "img",
    "aria-label": `${spec.title || "Bars"}. Values are listed in the table view.` }, host);
  for (const t of ticks) {
    svgEl("line", { x1: x(t), x2: x(t), y1: M.t, y2: H - M.b, class: "viz-grid" }, svg);
    const lab = svgEl("text", { x: x(t), y: H - M.b + 16, class: "viz-tick", "text-anchor": "middle" }, svg);
    lab.textContent = f(t).replace(" pp", "");
  }
  rows.forEach((r, i) => {
    const cy = M.t + i * rowH + rowH / 2;
    const lab = svgEl("text", { x: M.l - 10, y: cy, class: "viz-rowlabel", "text-anchor": "end", "dominant-baseline": "middle" }, svg);
    lab.textContent = r.label;
    if (r.value == null || !Number.isFinite(r.value)) return;
    const x0 = x(0), x1 = x(r.value), w = Math.abs(x1 - x0), left = Math.min(x0, x1);
    const rad = Math.min(4, w / 2);
    // 4px rounded data-end, square at the baseline
    const d = r.value >= 0
      ? `M${left},${cy - thick / 2}h${Math.max(w - rad, 0)}a${rad},${rad} 0 0 1 ${rad},${rad}v${thick - 2 * rad}a${rad},${rad} 0 0 1 ${-rad},${rad}h${-Math.max(w - rad, 0)}Z`
      : `M${left + w},${cy - thick / 2}h${-Math.max(w - rad, 0)}a${rad},${rad} 0 0 0 ${-rad},${rad}v${thick - 2 * rad}a${rad},${rad} 0 0 0 ${rad},${rad}h${Math.max(w - rad, 0)}Z`;
    svgEl("path", { d, style: `fill:${COLOR[r.color || "neutral"]}` }, svg);
    if (r.lo != null && r.hi != null) svgEl("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, class: "viz-whisker-ink" }, svg);
    const v = svgEl("text", { x: W - M.r + 8, y: cy, class: "viz-value", "dominant-baseline": "middle" }, svg);
    v.textContent = f(r.value);
  });
  svgEl("line", { x1: x(0), x2: x(0), y1: M.t, y2: H - M.b, class: "viz-axis" }, svg);
}

// ---------------------------------------------------------------- strip (multiverse distributions)
export function strip(host, spec) {
  host.replaceChildren();
  const rows = spec.rows;
  const W = Math.max(280, host.clientWidth || 640);
  const labelW = Math.min(spec.labelWidth || 210, W * 0.42);
  const rowH = 30, M = { t: 6, r: 118, b: 30, l: labelW + 10 };
  const H = M.t + rows.length * rowH + M.b;
  const all = rows.flatMap((r) => r.values);
  const dom = extent([0, ...all], 0.06);
  const ticks = niceTicks(dom[0], dom[1], 6);
  const x = linear(Math.min(dom[0], ticks[0]), Math.max(dom[1], ticks[ticks.length - 1]), M.l, W - M.r);
  const f = fmt[spec.format || "dec2"];
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz-svg", role: "img",
    "aria-label": `${spec.title || "Distribution of estimates"}. Summaries are listed in the table view.` }, host);
  for (const t of ticks) {
    svgEl("line", { x1: x(t), x2: x(t), y1: M.t, y2: H - M.b, class: "viz-grid" }, svg);
    const lab = svgEl("text", { x: x(t), y: H - M.b + 16, class: "viz-tick", "text-anchor": "middle" }, svg);
    lab.textContent = f(t);
  }
  svgEl("line", { x1: x(0), x2: x(0), y1: M.t, y2: H - M.b, class: "viz-axis" }, svg);
  if (spec.xLabel) {
    const t = svgEl("text", { x: W - M.r, y: H - 2, class: "viz-note", "text-anchor": "end" }, svg);
    t.textContent = spec.xLabel;
  }
  rows.forEach((r, i) => {
    const cy = M.t + i * rowH + rowH / 2;
    const lab = svgEl("text", { x: M.l - 10, y: cy, class: "viz-rowlabel", "text-anchor": "end", "dominant-baseline": "middle" }, svg);
    lab.textContent = r.label;
    // deterministic jitter so re-renders do not reshuffle points
    r.values.forEach((v, k) => {
      const j = ((Math.sin(k * 12.9898 + i * 78.233) * 43758.5453) % 1) * 0.36 * rowH;
      svgEl("circle", { cx: x(v), cy: cy + j, r: 2.2, style: `fill:${COLOR[spec.color || "women"]};opacity:.5` }, svg);
    });
    const s = [...r.values].sort((a, b) => a - b);
    if (s.length) {
      const med = s[Math.floor(s.length / 2)];
      svgEl("line", { x1: x(med), x2: x(med), y1: cy - 9, y2: cy + 9, class: "viz-whisker-ink" }, svg);
      const pos = s.filter((v) => v > 0).length / s.length;
      const t = svgEl("text", { x: W - M.r + 8, y: cy, class: "viz-value", "dominant-baseline": "middle" }, svg);
      t.textContent = `${Math.round(100 * pos)}% > 0 · n=${s.length}`;
    }
  });
}

// ---------------------------------------------------------------- mount
const KINDS = { line: lineChart, multiples: smallMultiples, dots: dotPlot, heatmap, bars, strip };

export function mountAll(root = document, overrides = null) {
  const figs = [...root.querySelectorAll("figure[data-viz]")].filter((f) => !f.closest("[hidden]"));
  for (const fig of figs) {
    const specEl = fig.querySelector("script[type='application/json']");
    const plot = fig.querySelector(".viz-plot");
    const draw = KINDS[fig.dataset.viz];
    if (!specEl || !plot || !draw) continue;
    let spec;
    try { spec = JSON.parse(specEl.textContent); } catch { continue; }
    if (overrides?.x && (fig.dataset.viz === "line" || fig.dataset.viz === "multiples")) {
      spec = { ...spec, x: { ...(spec.x || {}), ...overrides.x } };
      const [a, b] = overrides.x.domain;
      const crop = (s) => ({ ...s, points: s.points.filter((p) => p.x >= a && p.x <= b) });
      if (spec.series) spec.series = spec.series.map(crop);
      if (spec.panels) spec.panels = spec.panels.map((p) => ({ ...p, series: p.series.map(crop) }));
    }
    const render = () => draw(plot, spec);
    render();
    if ("ResizeObserver" in window) {
      let last = plot.clientWidth;
      new ResizeObserver(() => {
        if (Math.abs(plot.clientWidth - last) > 24) { last = plot.clientWidth; render(); }
      }).observe(plot);
    }
  }
}
