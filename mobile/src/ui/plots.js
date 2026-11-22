// Small canvas charts (histogram, scatter, line/step) with a touch read-out.
// Marks follow the dataviz conventions used across the app: thin bars with a
// 2px gap and rounded tops on a shared baseline, 2px lines, 8px markers with a
// surface ring, recessive grid, and text in ink colours (never the series colour).

import { h } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import htm from 'htm';
import { THEME, fmtNum } from './theme.js';

const html = htm.bind(h);

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function fmtAxis(v) {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e9) return `${Number((v / 1e9).toPrecision(3))}G`;
  if (a >= 1e6) return `${Number((v / 1e6).toPrecision(3))}M`;
  if (a >= 1e3) return `${Number((v / 1e3).toPrecision(3))}k`;
  if (a < 1e-2) return v.toExponential(1).replace('e-', 'e-');
  return String(Number(v.toPrecision(3)));
}

function extent(spec) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const see = (x, y) => {
    if (Number.isFinite(x)) { if (x < x0) x0 = x; if (x > x1) x1 = x; }
    if (Number.isFinite(y)) { if (y < y0) y0 = y; if (y > y1) y1 = y; }
  };
  if (spec.bars) {
    see(spec.bars.edges[0], 0); see(spec.bars.edges[spec.bars.edges.length - 1], 0);
    for (const c of spec.bars.counts) see(NaN, c);
  }
  for (const s of spec.series || []) for (const p of s.points) see(p.x, p.y);
  if (spec.yZero || spec.bars) y0 = Math.min(0, y0);
  if (spec.xZero) x0 = Math.min(0, x0);
  if (spec.xRange) { x0 = spec.xRange[0]; x1 = spec.xRange[1]; }
  if (!(x1 > x0)) { x0 -= 0.5; x1 += 0.5; }
  if (!(y1 > y0)) { y0 -= 0.5; y1 += 0.5; }
  const padY = (y1 - y0) * 0.08;
  y1 += padY;
  if (!(spec.yZero || spec.bars) && !spec.xyTight) y0 -= padY;
  if (!spec.bars && !spec.xRange) { const padX = (x1 - x0) * 0.04; x0 -= spec.xZero ? 0 : padX; x1 += padX; }
  return { x0, x1, y0, y1 };
}

function drawChart(ctx, W, H, spec, hover) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = THEME.surface;
  ctx.fillRect(0, 0, W, H);
  const padL = 44, padR = 12, padT = 10, padB = spec.xLabel ? 38 : 24;
  const pw = Math.max(10, W - padL - padR), ph = Math.max(10, H - padT - padB);
  const e = extent(spec);
  const X = (v) => padL + ((v - e.x0) / (e.x1 - e.x0)) * pw;
  const Y = (v) => padT + ph - ((v - e.y0) / (e.y1 - e.y0)) * ph;

  // grid + axes
  ctx.font = `11px ${THEME.sans}`;
  ctx.lineWidth = 1;
  ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  for (const t of niceTicks(e.y0, e.y1, Math.max(2, Math.floor(ph / 36)))) {
    const y = Math.round(Y(t)) + 0.5;
    ctx.strokeStyle = THEME.grid;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke();
    ctx.fillStyle = THEME.muted;
    ctx.fillText(fmtAxis(t), padL - 6, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const xt = niceTicks(e.x0, e.x1, Math.max(2, Math.floor(pw / 64)));
  for (const t of xt) {
    const x = Math.round(X(t)) + 0.5;
    ctx.strokeStyle = THEME.axis;
    ctx.beginPath(); ctx.moveTo(x, padT + ph); ctx.lineTo(x, padT + ph + 4); ctx.stroke();
    ctx.fillStyle = THEME.muted;
    ctx.fillText(fmtAxis(t), x, padT + ph + 6);
  }
  ctx.strokeStyle = THEME.axis;
  ctx.beginPath(); ctx.moveTo(padL, Math.round(Y(Math.max(e.y0, 0))) + 0.5); ctx.lineTo(padL + pw, Math.round(Y(Math.max(e.y0, 0))) + 0.5); ctx.stroke();
  if (spec.xLabel) {
    ctx.fillStyle = THEME.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(spec.xLabel, padL + pw / 2, H - 4);
  }
  if (spec.yLabel) {
    ctx.save();
    ctx.translate(11, padT + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = THEME.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(spec.yLabel, 0, 0);
    ctx.restore();
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(padL, padT - 2, pw, ph + 3); ctx.clip();
  const hits = [];

  // shaded bands (e.g. aggregation classes)
  for (const b of spec.bands || []) {
    const xa = Math.max(padL, X(b.x0)), xb = Math.min(padL + pw, X(b.x1));
    if (xb <= xa) continue;
    ctx.fillStyle = b.color; ctx.globalAlpha = 0.1;
    ctx.fillRect(xa, padT, xb - xa, ph);
    ctx.globalAlpha = 1;
  }

  if (spec.bars) {
    const { edges, counts, color } = spec.bars;
    const base = Y(0);
    for (let i = 0; i < counts.length; i++) {
      const xa = X(edges[i]), xb = X(edges[i + 1]);
      const gap = xb - xa > 5 ? 2 : xb - xa > 2.5 ? 1 : 0;
      const x = xa + gap / 2, w = Math.max(1, xb - xa - gap);
      hits.push({ x: (xa + xb) / 2, y: Y(counts[i]), bar: i, x0: xa, x1: xb });
      if (!counts[i]) continue;
      const top = Y(counts[i]);
      const r = Math.min(4, w / 2, Math.max(0, base - top));
      ctx.fillStyle = color || '#3987e5';
      ctx.globalAlpha = hover && hover.bar === i ? 1 : 0.86;
      ctx.beginPath();
      ctx.moveTo(x, base); ctx.lineTo(x, top + r);
      ctx.arcTo(x, top, x + r, top, r); ctx.lineTo(x + w - r, top);
      ctx.arcTo(x + w, top, x + w, top + r, r); ctx.lineTo(x + w, base);
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  (spec.series || []).forEach((s, si) => {
    const pts = s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (s.line !== false && pts.length > 1) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2;
      ctx.lineJoin = 'round';
      if (s.dash) ctx.setLineDash(s.dash);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = X(p.x), y = Y(p.y);
        if (i === 0) ctx.moveTo(x, y);
        else if (s.step) { ctx.lineTo(x, Y(pts[i - 1].y)); ctx.lineTo(x, y); }
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (s.markers) {
      const r = s.radius || 4;
      for (const p of pts) {
        ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r, 0, 6.2832);
        ctx.fillStyle = s.color; ctx.globalAlpha = s.alpha || 0.9; ctx.fill(); ctx.globalAlpha = 1;
        ctx.lineWidth = 1.5; ctx.strokeStyle = THEME.surface; ctx.stroke();
      }
    }
    if (s.hit !== false) pts.forEach((p, i) => hits.push({ x: X(p.x), y: Y(p.y), series: si, index: i, p }));
  });

  // vertical reference lines with a direct label
  for (const m of spec.marks || []) {
    const x = X(m.x);
    if (x < padL || x > padL + pw) continue;
    ctx.strokeStyle = m.color || THEME.ink2; ctx.lineWidth = 1.5;
    ctx.setLineDash(m.dash || [4, 3]);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
  for (const m of spec.marks || []) {
    if (!m.label) continue;
    const x = X(m.x);
    if (x < padL || x > padL + pw) continue;
    ctx.font = `600 10.5px ${THEME.sans}`; ctx.textBaseline = 'top';
    const tw = ctx.measureText(m.label).width;
    const left = x + tw + 8 > padL + pw;
    ctx.textAlign = left ? 'right' : 'left';
    ctx.fillStyle = THEME.ink2;
    ctx.fillText(m.label, left ? x - 4 : x + 4, padT + 2);
  }

  // hover read-out
  if (hover) {
    let lines = [];
    if (hover.bar !== undefined && spec.bars) {
      const { edges, counts } = spec.bars;
      lines = [`${fmtNum(edges[hover.bar], 3)} – ${fmtNum(edges[hover.bar + 1], 3)}`, `${counts[hover.bar]} ${spec.countLabel || 'particles'}`];
      ctx.strokeStyle = THEME.axis; ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(hover.x0) + 0.5, padT + 0.5, Math.round(hover.x1 - hover.x0), ph);
    } else if (hover.p) {
      const s = spec.series[hover.series];
      ctx.strokeStyle = THEME.axis; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(Math.round(hover.x) + 0.5, padT); ctx.lineTo(Math.round(hover.x) + 0.5, padT + ph); ctx.stroke();
      ctx.beginPath(); ctx.arc(hover.x, hover.y, 5.5, 0, 6.2832);
      ctx.fillStyle = s.color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = THEME.ink; ctx.stroke();
      lines = [
        `${spec.xName || 'x'} ${fmtNum(hover.p.x, 4)}${spec.xUnit ? ` ${spec.xUnit}` : ''}`,
        `${s.name || spec.yName || 'y'} ${fmtNum(hover.p.y, 4)}${spec.yUnit ? ` ${spec.yUnit}` : ''}`,
      ];
      if (hover.p.note) lines.push(hover.p.note);
    }
    if (lines.length) {
      ctx.font = `500 11.5px ${THEME.mono}`;
      const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14;
      const th = lines.length * 15 + 9;
      let bx = hover.x + 10, by = Math.max(padT, hover.y - th - 8);
      if (bx + tw > W - 4) bx = hover.x - tw - 10;
      bx = Math.max(4, bx);
      ctx.fillStyle = 'rgba(11,14,20,0.94)';
      ctx.strokeStyle = THEME.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.rect(bx, by, tw, th); ctx.fill(); ctx.stroke();
      ctx.fillStyle = THEME.ink; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      lines.forEach((l, i) => ctx.fillText(l, bx + 7, by + 5 + i * 15));
    }
  }
  return hits;
}

/**
 * @param {{spec: object, height?: number, onCanvas?: (c: HTMLCanvasElement) => void}} props
 */
export function Chart({ spec, height = 190, onCanvas }) {
  const ref = useRef(null);
  const hitsRef = useRef([]);
  const [hover, setHover] = useState(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => setWidth(Math.round(el.parentElement.getBoundingClientRect().width));
    measure();
    if (typeof ResizeObserver === 'undefined') { window.addEventListener('resize', measure); return () => window.removeEventListener('resize', measure); }
    const ro = new ResizeObserver(measure);
    ro.observe(el.parentElement);
    return () => ro.disconnect();
  }, []);

  useEffect(() => { setHover(null); }, [spec]);

  useEffect(() => {
    const c = ref.current;
    if (!c || !width) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    c.width = Math.round(width * dpr); c.height = Math.round(height * dpr);
    c.style.width = `${width}px`; c.style.height = `${height}px`;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hitsRef.current = drawChart(ctx, width, height, spec, hover);
    if (onCanvas) onCanvas(c);
  }, [spec, hover, width, height]);

  const pick = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = Infinity;
    for (const hit of hitsRef.current) {
      const d = hit.bar !== undefined ? (x >= hit.x0 && x <= hit.x1 ? 0 : Math.abs(hit.x - x) + 1000) : Math.hypot(hit.x - x, (hit.y - y) * 0.6);
      if (d < bd) { bd = d; best = hit; }
    }
    setHover(best && (best.bar !== undefined ? bd === 0 : bd < 60) ? best : null);
  };

  return html`<div class="chart" style=${{ height: `${height}px` }}>
    <canvas ref=${ref} role="img" aria-label=${spec.aria || spec.title || 'Chart'}
      onPointerDown=${pick} onPointerMove=${(e) => { if (e.pointerType === 'mouse' || e.buttons) pick(e); }}
      onPointerLeave=${(e) => { if (e.pointerType === 'mouse') setHover(null); }}></canvas>
  </div>`;
}

export function Legend({ items }) {
  if (!items || items.length < 2) return null;
  return html`<div class="legend">${items.map((it) => html`<span class="legend-item" key=${it.name}>
    <span class=${`legend-swatch${it.line ? ' is-line' : ''}`} style=${{ background: it.color }}></span>${it.name}</span>`)}</div>`;
}
