// Small reusable controls.

import { h } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import htm from 'htm';
import { THEME, fmtNum } from './theme.js';

export const html = htm.bind(h);

const ICONS = {
  files: 'M4 6.5A1.5 1.5 0 0 1 5.5 5h4l2 2.5h7A1.5 1.5 0 0 1 20 9v8.5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z',
  display: 'M4 7h10M18 7h2M4 12h3M11 12h9M4 17h12M20 17h0M16 5v4M9 10v4M18 15v4',
  fit: 'M12 3v4M12 17v4M3 12h4M17 12h4M12 12h.01M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10z',
  layers: 'M12 4l8 4-8 4-8-4zM4 12l8 4 8-4M4 16l8 4 8-4',
  movie: 'M5 5h14a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM8 5v14M16 5v14M4 9.5h4M4 14.5h4M16 9.5h4M16 14.5h4',
  play: 'M8 5.5v13l11-6.5z',
  pause: 'M7.5 5.5h3v13h-3zM13.5 5.5h3v13h-3z',
  prev: 'M15.5 6l-7 6 7 6M7 6v12',
  next: 'M8.5 6l7 6-7 6M17 6v12',
  close: 'M6 6l12 12M18 6L6 18',
  plus: 'M12 5v14M5 12h14',
  expand: 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5',
  eye: 'M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z',
  eyeoff: 'M3 3l18 18M10.6 5.7A9 9 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a16 16 0 0 1-3 3.7M6.5 7.4A15.600 15.600 0 0 0 2.500 12s3.500 6.500 9.500 6.500c1.400 0 2.700-.3 3.800-.9',
  save: 'M12 4v11M7.5 10.500L12 15l4.500-4.500M5 19h14',
  drive: 'M9 4h6l6 10.500-3 5H6l-3-5zM9 4l6 10.500M15 4L3.200 14.500M6 19.500l3-5h12',
  chevron: 'M8 10l4 4 4-4',
  up: 'M12 19V6M6.500 11.500L12 6l5.500 5.500',
  down: 'M12 5v13M6.500 12.500L12 18l5.500-5.500',
  left: 'M19 12H6M11.500 6.500L6 12l5.500 5.500',
  right: 'M5 12h13M12.500 6.500L18 12l-5.500 5.500',
  trash: 'M5 7h14M10 7V4.500h4V7M7 7l1 12.500h8L17 7M10.500 10.500v6M13.500 10.500v6',
  info: 'M12 8h.01M11 11.500h1v5h1M12 3.500a8.500 8.500 0 1 0 0 17 8.500 8.500 0 0 0 0-17z',
  folder: 'M4 7a1.500 1.500 0 0 1 1.500-1.500h4l2 2.500h7A1.500 1.500 0 0 1 20 9.500v8A1.500 1.500 0 0 1 18.500 19h-13A1.500 1.500 0 0 1 4 17.500z',
  image: 'M5 5h14a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM4 15.500l4.500-4.500 4 4 3-3 4.500 4.500M15.500 9.500h.01',
  check: 'M5 12.500l4.500 4.500L19 7.500',
  search: 'M10.500 4.500a6 6 0 1 0 0 12 6 6 0 0 0 0-12zM15 15l5 5',
  chart: 'M4 20V4M4 20h16M8 20v-6M12 20V8M16 20v-9',
  track: 'M4 18c3-1 2-6 5-7s3 4 6 2 2-7 5-8M4 18h.01M20 5h.01',
};

export function Icon({ name, size = 20, fill = false }) {
  const d = ICONS[name] || ICONS.info;
  const filled = fill || name === 'play' || name === 'pause';
  return html`<svg class="icon" width=${size} height=${size} viewBox="0 0 24 24" aria-hidden="true"
    fill=${filled ? 'currentColor' : 'none'} stroke=${filled ? 'none' : 'currentColor'}
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d=${d}></path></svg>`;
}

export function Btn({ children, onClick, kind = '', icon, disabled, title, small, id }) {
  return html`<button type="button" id=${id} class=${`btn ${kind} ${small ? 'btn-small' : ''}`} onClick=${onClick}
    disabled=${disabled} title=${title} aria-label=${!children && title ? title : undefined}>
    ${icon ? html`<${Icon} name=${icon} size=${small ? 16 : 18} />` : null}${children ? html`<span>${children}</span>` : null}</button>`;
}

export function IconBtn({ icon, onClick, title, active, disabled, id, size = 20 }) {
  return html`<button type="button" id=${id} class=${`icon-btn${active ? ' is-active' : ''}`} onClick=${onClick} title=${title}
    aria-label=${title} aria-pressed=${active === undefined ? undefined : String(!!active)} disabled=${disabled}>
    <${Icon} name=${icon} size=${size} /></button>`;
}

export function Toggle({ label, checked, onChange, id, hint }) {
  return html`<label class="toggle" for=${id}>
    <span class="toggle-text"><span>${label}</span>${hint ? html`<small>${hint}</small>` : null}</span>
    <input id=${id} type="checkbox" role="switch" checked=${!!checked} onChange=${(e) => onChange(e.currentTarget.checked)} />
    <span class="toggle-track" aria-hidden="true"><span class="toggle-thumb"></span></span>
  </label>`;
}

export function Seg({ options, value, onChange, label, id }) {
  return html`<div class="seg" role="group" aria-label=${label} id=${id}>
    ${options.map((o) => html`<button type="button" key=${o.value} class=${`seg-item${o.value === value ? ' is-on' : ''}`}
      aria-pressed=${String(o.value === value)} disabled=${o.disabled} onClick=${() => onChange(o.value)}>${o.label}</button>`)}
  </div>`;
}

/** Number field that keeps what the user is typing and commits valid values. */
export function Num({ label, value, onChange, min, max, step = 'any', unit, id, digits = 4, width }) {
  const [text, setText] = useState(null);
  const shown = text !== null ? text : (Number.isFinite(value) ? String(Number(value.toPrecision(digits))) : '');
  const commit = (t) => {
    const v = parseFloat(String(t).replace(',', '.'));
    if (!Number.isFinite(v)) return;
    let c = v;
    if (min !== undefined) c = Math.max(min, c);
    if (max !== undefined) c = Math.min(max, c);
    onChange(c);
  };
  return html`<label class="field" for=${id} style=${width ? { width } : null}>
    ${label ? html`<span class="field-label">${label}</span>` : null}
    <span class="field-box">
      <input id=${id} type="text" inputmode="decimal" autocomplete="off" autocapitalize="off" spellcheck=${false} value=${shown}
        onInput=${(e) => { setText(e.currentTarget.value); commit(e.currentTarget.value); }}
        onBlur=${() => setText(null)} onKeyDown=${(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }} />
      ${unit ? html`<span class="field-unit">${unit}</span>` : null}
    </span>
  </label>`;
}

export function Slider({ label, value, min, max, step, onChange, format, id, hint }) {
  return html`<div class="slider">
    <div class="slider-head"><label for=${id}>${label}</label><output for=${id}>${format ? format(value) : fmtNum(value)}</output></div>
    <input id=${id} type="range" min=${min} max=${max} step=${step} value=${value}
      onInput=${(e) => onChange(parseFloat(e.currentTarget.value))} />
    ${hint ? html`<small class="hint">${hint}</small>` : null}
  </div>`;
}

export function Stepper({ label, value, min, max, step = 1, onChange, id, format }) {
  const set = (v) => onChange(Math.min(max, Math.max(min, Number((v).toFixed(6)))));
  return html`<div class="stepper">
    <span class="field-label" id=${`${id}-label`}>${label}</span>
    <div class="stepper-box" role="group" aria-labelledby=${`${id}-label`}>
      <button type="button" aria-label=${`Decrease ${label}`} disabled=${value <= min} onClick=${() => set(value - step)}>−</button>
      <output id=${id}>${format ? format(value) : value}</output>
      <button type="button" aria-label=${`Increase ${label}`} disabled=${value >= max} onClick=${() => set(value + step)}>+</button>
    </div>
  </div>`;
}

export function Section({ title, children, aside, id }) {
  return html`<section class="section" id=${id}>
    ${title ? html`<div class="section-head"><h3>${title}</h3>${aside || null}</div>` : null}
    ${children}
  </section>`;
}

/** Key/value rows with aligned numbers. */
export function Facts({ rows }) {
  return html`<dl class="facts">${rows.filter(Boolean).map(([k, v]) => html`<div class="fact" key=${k}><dt>${k}</dt><dd>${v}</dd></div>`)}</dl>`;
}

/**
 * Pixel-intensity histogram with draggable black and white points.
 * The x axis is compressed (asinh) so the background and the few bright
 * particle pixels are both usable.
 */
export function ContrastBar({ stats, lo, hi, onChange, color = '#aab3c5' }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const [width, setWidth] = useState(0);
  const H = 64;
  const span = Math.max(stats.max - stats.min, 1e-9);
  const knee = Math.max((stats.percentile(99) - stats.percentile(1)) * 0.6, span * 1e-4);
  const fwd = (v) => Math.asinh((v - stats.min) / knee) / Math.asinh(span / knee);
  const inv = (t) => stats.min + knee * Math.sinh(t * Math.asinh(span / knee));

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => setWidth(Math.round(el.parentElement.getBoundingClientRect().width));
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el.parentElement);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = ref.current;
    if (!c || !width) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    c.width = Math.round(width * dpr); c.height = Math.round(H * dpr);
    c.style.width = `${width}px`; c.style.height = `${H}px`;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, width, H);
    const pad = 10, pw = width - 2 * pad;
    const cols = Math.max(8, Math.floor(pw / 3));
    const sums = new Float64Array(cols);
    const bins = stats.bins;
    for (let b = 0; b < bins.length; b++) {
      if (!bins[b]) continue;
      const v = stats.min + ((b + 0.5) / bins.length) * span;
      sums[Math.min(cols - 1, Math.floor(fwd(v) * cols))] += bins[b];
    }
    let top = 0;
    for (let i = 0; i < cols; i++) top = Math.max(top, Math.log1p(sums[i]));
    const xl = pad + fwd(Math.min(Math.max(lo, stats.min), stats.max)) * pw;
    const xh = pad + fwd(Math.min(Math.max(hi, stats.min), stats.max)) * pw;
    for (let i = 0; i < cols; i++) {
      if (!sums[i]) continue;
      const x = pad + (i / cols) * pw, bh = Math.max(1.5, (Math.log1p(sums[i]) / (top || 1)) * (H - 20));
      const mid = x + pw / cols / 2;
      ctx.fillStyle = color;
      ctx.globalAlpha = mid >= xl && mid <= xh ? 0.9 : 0.3;
      ctx.fillRect(x, H - 12 - bh, Math.max(1, pw / cols - 1), bh);
    }
    ctx.globalAlpha = 1;
    for (const [x, col] of [[xl, '#ffffff'], [xh, THEME.accent]]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, H - 12); ctx.stroke();
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(x - 6, H); ctx.lineTo(x + 6, H); ctx.lineTo(x, H - 11); ctx.closePath(); ctx.fill();
    }
  }, [stats, lo, hi, width, color]);

  const at = (e) => {
    const r = ref.current.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (e.clientX - r.left - 10) / Math.max(1, r.width - 20)));
    return inv(t);
  };
  const move = (e) => {
    if (!drag.current) return;
    const v = at(e);
    const eps = span * 1e-5;
    if (drag.current === 'lo') onChange(Math.min(v, hi - eps), hi); else onChange(lo, Math.max(v, lo + eps));
  };
  return html`<div class="contrast">
    <canvas ref=${ref} aria-label="Pixel intensity histogram with black and white points"
      onPointerDown=${(e) => {
        const v = at(e);
        drag.current = Math.abs(fwd(v) - fwd(lo)) <= Math.abs(fwd(v) - fwd(hi)) ? 'lo' : 'hi';
        try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        move(e);
      }}
      onPointerMove=${move} onPointerUp=${() => { drag.current = null; }} onPointerCancel=${() => { drag.current = null; }}></canvas>
  </div>`;
}
