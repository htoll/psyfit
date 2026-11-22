// Colours shared by canvas drawing (viewer, plots) and number formatting.
// The CSS custom properties in app.css carry the same values.

export const THEME = {
  canvas: '#07090d',
  bg: '#0b0e14',
  surface: '#141925',
  surface2: '#1c2333',
  line: '#2a3347',
  grid: 'rgba(255,255,255,0.07)',
  axis: 'rgba(255,255,255,0.22)',
  ink: '#e8ecf4',
  ink2: '#b4bccd',
  muted: '#8d97ab',
  accent: '#f2b84b',
  sans: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  mono: 'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace',
};

/** 532 pps / 12.4 kpps / 1.24 Mpps. `short` drops the unit to "12.4k" for map labels. */
export function fmtBrightness(v, short = false) {
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 1 : 2)}${short ? 'M' : ' Mpps'}`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e5 ? 0 : a >= 1e4 ? 1 : 2)}${short ? 'k' : ' kpps'}`;
  return `${v.toFixed(a >= 100 ? 0 : 1)}${short ? '' : ' pps'}`;
}

/** Compact number with about `digits` significant figures (1.23e4 style for big/small). */
export function fmtNum(v, digits = 3) {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(0, digits - 1)).replace('e+', 'e');
  return String(Number(v.toPrecision(digits)));
}

export function fmtBytes(n) {
  if (!Number.isFinite(n)) return '';
  if (n >= 1073741824) return `${(n / 1073741824).toFixed(2)} GB`;
  if (n >= 1048576) return `${(n / 1048576).toFixed(n >= 104857600 ? 0 : 1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${n} B`;
}

export function fmtDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export function fmtDuration(s) {
  if (!Number.isFinite(s)) return '–';
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 100) return `${Number(s.toPrecision(3))} s`;
  return `${(s / 60).toFixed(1)} min`;
}
