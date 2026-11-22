// Nikon spectra: from a localization to its emission spectrum.
//
// A Nikon frame holds the direct image of the field at the top left (the
// "localization" channel, PsyFit's 280 px "Mr Beam" region) and, at the bottom
// right, the same field dispersed by wavelength: every particle appears there
// as a short horizontal streak. A calibration measured on the microscope says
// where the streak of a particle at a given position lands, and which
// wavelength each pixel along it is.
//
// This is a port of PsyFit's Get Spectra (tools/get_spectra.py and
// utils.get_image), step for step, so spectra agree with the desktop tool.
// PsyFit works on the frame flipped top to bottom; "flipped" coordinates below
// are (x = column, y = H - 1 - row), and the localization of a fit at sensor
// position (x, y) is the object point (x, H - y), as PsyFit computes it.

import { unpickle, PickleError } from './pickle.js';

/** The spectrum is sampled from SPEC_LEFT pixels left of its anchor to SPEC_RIGHT right of it. */
export const SPEC_LEFT = 90;
export const SPEC_RIGHT = 9;
/** Rows averaged above and below the anchor row (5 rows in all). */
export const SPEC_HALF_ROWS = 2;
export const SPEC_NPTS = 100;

export class CalibrationError extends Error {
  constructor(message) { super(message); this.name = 'CalibrationError'; }
}

const pyRound = (v) => {            // Python's round(): half to even
  const f = Math.floor(v), d = v - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
};

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/** "saving_info" (the grid) or "fits" (the wavelength fits), as PsyFit tells them apart by name. */
export function calibrationKind(filename) {
  const name = String(filename || '').toLowerCase();
  if (name.includes('saving_info')) return 'grid';
  if (name.includes('fits')) return 'fits';
  return null;
}

/** The six-digit date in a calibration file name ("saving_info_261116_no_tracking.pkl" -> "261116"). */
export function calibrationDate(filename) {
  const m = /(?:^|[^0-9])(\d{6})(?:[^0-9]|$)/.exec(String(filename || ''));
  return m ? m[1] : '';
}

const pair = (v) => (v && v.length >= 2 && Number.isFinite(Number(v[0])) && Number.isFinite(Number(v[1])) ? [Number(v[0]), Number(v[1])] : null);

/**
 * Build a calibration from the two unpickled files.
 * @param {Map} grid "saving_info": id -> [label, (x, y) object point, spot fit, (xs, ys) spectral point, spectrum]
 * @param {Map} fits id -> (a, b, c) with wavelength(i) = a * exp(b * i) + c for sample i of the window
 */
export function buildCalibration(grid, fits, meta = {}) {
  if (!(grid instanceof Map) || !grid.size) throw new CalibrationError('The saving_info file holds no calibration points');
  if (!(fits instanceof Map) || !fits.size) throw new CalibrationError('The fits file holds no wavelength fits');
  const ids = [], obj = [], img = [], amp = [], fit = [];
  for (const [id, v] of grid) {
    const o = Array.isArray(v) ? pair(v[1]) : null, s = Array.isArray(v) ? pair(v[3]) : null;
    if (!o || !s) continue;
    const f = fits.get(id);
    const ok = f && f.length >= 3 && [f[0], f[1], f[2]].every((x) => Number.isFinite(Number(x)));
    ids.push(String(id)); obj.push(o[0], o[1]); img.push(s[0], s[1]);
    amp.push(v[2] && Number.isFinite(Number(v[2][0])) ? Number(v[2][0]) : NaN);
    fit.push(ok ? Number(f[0]) : NaN, ok ? Number(f[1]) : NaN, ok ? Number(f[2]) : NaN);
  }
  if (ids.length < 3) throw new CalibrationError('This saving_info file is not a spectral calibration grid (use the "..._no_tracking.pkl" file)');
  let withFit = 0;
  for (let i = 0; i < ids.length; i++) if (Number.isFinite(fit[i * 3])) withFit++;
  if (!withFit) throw new CalibrationError('The fits file does not belong to this saving_info file (no point has a wavelength fit)');
  return {
    name: meta.name || '', fitsName: meta.fitsName || '', date: meta.date || calibrationDate(meta.name), demo: !!meta.demo,
    n: ids.length, withFit, ids, obj: Float64Array.from(obj), img: Float64Array.from(img), amp: Float64Array.from(amp), fit: Float64Array.from(fit),
  };
}

/** Read the two .pkl files of a calibration. `files`: [{name, bytes}] in any order. */
export function readCalibrationFiles(files) {
  let grid = null, fits = null;
  for (const f of files) {
    const kind = calibrationKind(f.name);
    if (kind === 'grid' && !grid) grid = f; else if (kind === 'fits' && !fits) fits = f;
  }
  if (!grid || !fits) throw new CalibrationError('A calibration is two files: one whose name contains "saving_info" and one whose name contains "fits"');
  const load = (f) => {
    try { return unpickle(f.bytes); } catch (err) {
      if (err instanceof PickleError) throw new CalibrationError(`${f.name}: ${err.message}`);
      throw err;
    }
  };
  return buildCalibration(load(grid), load(fits), { name: grid.name, fitsName: fits.name });
}

/** Plain-JSON form of a calibration (for storing on the device) and back. */
export function calibrationToJson(cal) {
  return { v: 1, name: cal.name, fitsName: cal.fitsName, date: cal.date, ids: cal.ids, obj: Array.from(cal.obj), img: Array.from(cal.img), amp: Array.from(cal.amp, (v) => (Number.isFinite(v) ? v : null)), fit: Array.from(cal.fit, (v) => (Number.isFinite(v) ? v : null)) };
}
export function calibrationFromJson(j) {
  if (!j || j.v !== 1 || !Array.isArray(j.ids) || !Array.isArray(j.obj) || j.obj.length !== j.ids.length * 2) return null;
  const nan = (a) => Float64Array.from(a, (v) => (v === null ? NaN : v));
  const fit = nan(j.fit);
  let withFit = 0;
  for (let i = 0; i < j.ids.length; i++) if (Number.isFinite(fit[i * 3])) withFit++;
  return { name: j.name || '', fitsName: j.fitsName || '', date: j.date || '', demo: false, n: j.ids.length, withFit, ids: j.ids, obj: Float64Array.from(j.obj), img: Float64Array.from(j.img), amp: nan(j.amp), fit };
}

// ---------------------------------------------------------------------------
// Localization -> spectrum
// ---------------------------------------------------------------------------

/** Object point of a fit at sensor (x, y) in a frame H rows tall, in PsyFit's flipped coordinates. */
export const objectPoint = (x, y, H) => [x, H - y];

/** Calibration points ordered by distance from (x, y). */
function nearest(cal, x, y) {
  const d = new Float64Array(cal.n);
  for (let i = 0; i < cal.n; i++) d[i] = Math.hypot(x - cal.obj[i * 2], y - cal.obj[i * 2 + 1]);
  return Array.from({ length: cal.n }, (_, i) => i).sort((a, b) => (d[a] - d[b]) || (a - b));
}

/**
 * Where the spectrum of a particle at object point (x, y) lands: utils.get_image.
 * The three nearest calibration points whose directions from the first are
 * close to perpendicular span a local frame; the particle's coordinates in it
 * carry over to the same three points' spectral positions.
 * @returns {null | [number, number]} (xs, ys) in flipped coordinates
 */
export function mapToSpectrum(cal, x, y) {
  const order = nearest(cal, x, y);
  if (order.length < 3) return null;
  const P = (i) => [cal.obj[i * 2], cal.obj[i * 2 + 1]];
  const g = [order[0], order[1], order[2]];
  const cosAngle = () => {
    const a = P(g[0]), b = P(g[1]), c = P(g[2]);
    const u = [b[0] - a[0], b[1] - a[1]], v = [c[0] - a[0], c[1] - a[1]];
    return (u[0] * v[0] + u[1] * v[1]) / Math.hypot(u[0], u[1]) / Math.hypot(v[0], v[1]);
  };
  let k = 3, cos = cosAngle();
  while (Math.abs(cos) > 0.1 || cos !== cos) {
    if (k >= order.length) return null;
    g[2] = order[k++];
    cos = cosAngle();
  }
  const a = P(g[0]), b = P(g[1]), c = P(g[2]);
  // solve (x, y) - a = c0 * (b - a) + c1 * (c - a)
  const m00 = b[0] - a[0], m01 = c[0] - a[0], m10 = b[1] - a[1], m11 = c[1] - a[1];
  const det = m00 * m11 - m01 * m10;
  if (!(Math.abs(det) > 1e-12)) return null;
  const rx = x - a[0], ry = y - a[1];
  const c0 = (rx * m11 - m01 * ry) / det, c1 = (m00 * ry - rx * m10) / det;
  const I = (i) => [cal.img[i * 2], cal.img[i * 2 + 1]];
  const A = I(g[0]), B = I(g[1]), C = I(g[2]);
  return [A[0] + c0 * (B[0] - A[0]) + c1 * (C[0] - A[0]), A[1] + c0 * (B[1] - A[1]) + c1 * (C[1] - A[1])];
}

/**
 * Wavelength of each of the SPEC_NPTS samples for a particle at object point
 * (x, y): the exponential fit of the nearest calibration point that has one.
 * @returns {null | {nm: Float64Array, nmPerSample: Float64Array, id: string}}
 */
export function wavelengthAxis(cal, x, y) {
  const at = nearest(cal, x, y).find((i) => Number.isFinite(cal.fit[i * 3]));
  if (at === undefined) return null;
  const a = cal.fit[at * 3], b = cal.fit[at * 3 + 1], c = cal.fit[at * 3 + 2];
  const f = (i) => a * Math.exp(b * i) + c;
  const nm = new Float64Array(SPEC_NPTS), nmPerSample = new Float64Array(SPEC_NPTS);
  for (let i = 0; i < SPEC_NPTS; i++) { nm[i] = f(i); nmPerSample[i] = f(i + 0.5) - f(i - 0.5); }
  return { nm, nmPerSample, id: cal.ids[at] };
}

/**
 * The spectrum at spectral position (xs, ys): SPEC_NPTS samples from
 * xs - SPEC_LEFT to xs + SPEC_RIGHT, linearly interpolated along the row and
 * averaged over the rows within SPEC_HALF_ROWS of ys (fewer at the frame's
 * last rows, as PsyFit does). Null when ys is too close to the frame's edge.
 * @param {Float32Array} data frame, row-major, row 0 first (not flipped)
 */
export function sampleSpectrum(data, W, H, xs, ys) {
  const r = pyRound(ys);
  let rows;
  if (!(r >= SPEC_HALF_ROWS) || r > H - 1) return null;
  if (r < H - 2) rows = [-2, -1, 0, 1, 2];
  else if (r === H - 2) rows = [-2, -1, 0, 1];
  else rows = [-2, -1, 0];
  const out = new Float64Array(SPEC_NPTS);
  const start = xs - SPEC_LEFT, stop = xs + SPEC_RIGHT, step = (stop - start) / (SPEC_NPTS - 1);
  for (const j of rows) {
    const base = (H - 1 - (r + j)) * W;                 // flipped row r + j
    for (let i = 0; i < SPEC_NPTS; i++) {
      const xq = i === SPEC_NPTS - 1 ? stop : start + i * step;
      let v;
      if (!(xq > 0)) v = data[base];
      else if (xq >= W - 1) v = data[base + W - 1];
      else { const k = Math.floor(xq); v = data[base + k] + (xq - k) * (data[base + k + 1] - data[base + k]); }
      out[i] += v;
    }
  }
  for (let i = 0; i < SPEC_NPTS; i++) out[i] /= rows.length;
  return out;
}

/** PsyFit's test for two spectra that may run into each other: windows within reach in x and sharing rows. */
export function spectraOverlap(a, b) {
  return Math.abs(a[0] - b[0]) < SPEC_LEFT + SPEC_RIGHT && Math.abs(a[1] - b[1]) <= 2 * SPEC_HALF_ROWS;
}

/** True when the sampling window starts left of the detector (PsyFit drops such spectra). */
export const offDetector = (xs) => xs - SPEC_LEFT < 0;

/**
 * The sampling window in sensor coordinates (for drawing it on the image):
 * columns x0..x1 and the rows it averages, row 0 first.
 */
export function spectralWindow(xs, ys, H) {
  const r = pyRound(ys);
  return { x0: xs - SPEC_LEFT, x1: xs + SPEC_RIGHT, y0: H - 1 - (r + SPEC_HALF_ROWS), y1: H - 1 - (r - SPEC_HALF_ROWS) };
}
