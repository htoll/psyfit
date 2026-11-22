// Affine transforms, channel registration from matched PSFs, aperture (field
// stop) circle fitting and nearest-neighbour colocalization matching.
//
// An affine is [a, b, c, d, e, f]:  x' = a*x + b*y + c,  y' = d*x + e*y + f.

import { gaussianFilter } from './numeric.js';
import { mulberry32 } from './stats.js';

export const IDENTITY = Object.freeze([1, 0, 0, 0, 1, 0]);

export const translation = (dx, dy) => [1, 0, dx, 0, 1, dy];

export function applyAffine(m, x, y) {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

/** m2 after m1. */
export function composeAffine(m2, m1) {
  return [
    m2[0] * m1[0] + m2[1] * m1[3], m2[0] * m1[1] + m2[1] * m1[4], m2[0] * m1[2] + m2[1] * m1[5] + m2[2],
    m2[3] * m1[0] + m2[4] * m1[3], m2[3] * m1[1] + m2[4] * m1[4], m2[3] * m1[2] + m2[4] * m1[5] + m2[5],
  ];
}

export function invertAffine(m) {
  const det = m[0] * m[4] - m[1] * m[3];
  if (!(Math.abs(det) > 1e-12)) return null;
  const a = m[4] / det, b = -m[1] / det, d = -m[3] / det, e = m[0] / det;
  return [a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])];
}

export function isIdentity(m, tol = 1e-9) {
  return Math.abs(m[0] - 1) < tol && Math.abs(m[1]) < tol && Math.abs(m[2]) < tol
    && Math.abs(m[3]) < tol && Math.abs(m[4] - 1) < tol && Math.abs(m[5]) < tol;
}

/** Shift, scale, rotation and shear of an affine (skimage AffineTransform conventions). */
export function describeAffine(m) {
  const sx = Math.hypot(m[0], m[3]);
  const sy = Math.hypot(m[1], m[4]);
  const rotation = Math.atan2(m[3], m[0]);
  const shear = Math.atan2(-m[1], m[4]) - rotation;
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  return { tx: m[2], ty: m[5], scaleX: sx, scaleY: sy, rotationDeg: rotation * 180 / Math.PI, shearDeg: wrap(shear) * 180 / Math.PI };
}

function solve3(A, b) {
  const M = [A.slice(0, 3).concat(b[0]), A.slice(3, 6).concat(b[1]), A.slice(6, 9).concat(b[2])];
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
    }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

/** Least-squares affine taking src points onto dst points; null if degenerate. */
export function fitAffine(src, dst) {
  const n = Math.min(src.length, dst.length);
  if (n < 3) return null;
  // Centre for conditioning.
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += src[i][0]; my += src[i][1]; }
  mx /= n; my /= n;
  const A = new Array(9).fill(0), bx = [0, 0, 0], by = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const r = [src[i][0] - mx, src[i][1] - my, 1];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) A[a * 3 + b] += r[a] * r[b];
      bx[a] += r[a] * dst[i][0];
      by[a] += r[a] * dst[i][1];
    }
  }
  const px = solve3(A.slice(), bx), py = solve3(A.slice(), by);
  if (!px || !py) return null;
  return [px[0], px[1], px[2] - px[0] * mx - px[1] * my, py[0], py[1], py[2] - py[0] * mx - py[1] * my];
}

function nearestIndex(pts, x, y) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - x, pts[i][1] - y);
    if (d < bd) { bd = d; best = i; }
  }
  return [best, bd];
}

export const MIN_REG_MATCHES = 5;

/**
 * Affine map from `src` to `dst` point coordinates, estimated from PSF
 * positions that appear in both (PsyFit's _estimate_registration):
 * every src->dst displacement within `maxShift` votes in a 1-px histogram whose
 * peak is the coarse shift; points are paired after that shift; RANSAC fits an
 * affine and rejects mismatches; inliers are re-matched and refitted. Falls back
 * to the pure shift with few matches and returns null below `minMatches`
 * (default MIN_REG_MATCHES: with fewer common PSFs the vote can be a chance alignment).
 *
 * @param {Array<{src: number[][], dst: number[][]}>} sets one entry per image pair
 * @returns {null | {m: number[], kind: 'affine'|'shift', n: number, rms: number}}
 */
export function estimateRegistration(sets, opts = {}) {
  const maxShift = opts.maxShift ?? 30;
  const tol = opts.tol ?? 1.5;
  const minMatches = opts.minMatches ?? MIN_REG_MATCHES;
  const usable = sets.filter((s) => s.src.length >= 3 && s.dst.length >= 3);
  if (!usable.length) return null;

  const votes = [];
  for (const { src, dst } of usable) {
    for (const p of src) {
      for (const q of dst) {
        const dx = q[0] - p[0], dy = q[1] - p[1];
        if (dx * dx + dy * dy <= maxShift * maxShift) votes.push([dx, dy]);
      }
    }
  }
  if (!votes.length) return null;
  const nb = 2 * maxShift;
  const hist = new Float32Array(nb * nb);
  for (const [dx, dy] of votes) {
    let ix = Math.floor(dx + maxShift), iy = Math.floor(dy + maxShift);
    if (ix === nb) ix = nb - 1;
    if (iy === nb) iy = nb - 1;
    if (ix >= 0 && ix < nb && iy >= 0 && iy < nb) hist[iy * nb + ix] += 1;
  }
  const sm = gaussianFilter(hist, nb, nb, 1.0);
  let arg = 0;
  for (let i = 1; i < sm.length; i++) if (sm[i] > sm[arg]) arg = i;
  const coarse = [(arg % nb) - maxShift + 0.5, Math.floor(arg / nb) - maxShift + 0.5];
  let sx = 0, sy = 0, cnt = 0;
  for (const [dx, dy] of votes) {
    if (Math.hypot(dx - coarse[0], dy - coarse[1]) <= tol) { sx += dx; sy += dy; cnt++; }
  }
  const shift = cnt ? [sx / cnt, sy / cnt] : coarse;

  const match = (m, radius) => {
    const a = [], b = [];
    for (const { src, dst } of usable) {
      for (const p of src) {
        const [tx, ty] = applyAffine(m, p[0], p[1]);
        const [j, d] = nearestIndex(dst, tx, ty);
        if (j >= 0 && d <= radius) { a.push(p); b.push(dst[j]); }
      }
    }
    return [a, b];
  };

  const shiftM = translation(shift[0], shift[1]);
  let [a, b] = match(shiftM, 2 * tol);
  if (a.length < minMatches) return null;
  let result = { m: shiftM, kind: 'shift' };
  if (a.length >= 6 && opts.shiftOnly !== true) {
    const rand = mulberry32(0);
    let bestInliers = 0, bestModel = null;
    const trials = Math.min(2000, opts.maxTrials ?? 2000);
    for (let t = 0; t < trials; t++) {
      const i0 = Math.floor(rand() * a.length);
      let i1 = Math.floor(rand() * a.length), i2 = Math.floor(rand() * a.length);
      if (i1 === i0 || i2 === i0 || i1 === i2) continue;
      const m = fitAffine([a[i0], a[i1], a[i2]], [b[i0], b[i1], b[i2]]);
      if (!m) continue;
      let inl = 0;
      for (let i = 0; i < a.length; i++) {
        const [tx, ty] = applyAffine(m, a[i][0], a[i][1]);
        if (Math.hypot(tx - b[i][0], ty - b[i][1]) < tol) inl++;
      }
      if (inl > bestInliers) { bestInliers = inl; bestModel = m; if (inl === a.length) break; }
    }
    if (bestModel && bestInliers >= 6) {
      const ia = [], ib = [];
      for (let i = 0; i < a.length; i++) {
        const [tx, ty] = applyAffine(bestModel, a[i][0], a[i][1]);
        if (Math.hypot(tx - b[i][0], ty - b[i][1]) < tol) { ia.push(a[i]); ib.push(b[i]); }
      }
      const model = fitAffine(ia, ib) || bestModel;
      const [ra, rb] = match(model, tol);
      const refined = ra.length >= 6 ? fitAffine(ra, rb) : null;
      if (refined) result = { m: refined, kind: 'affine' };
    }
  }
  [a, b] = match(result.m, tol);
  if (a.length < minMatches) return null;
  let ss = 0;
  for (let i = 0; i < a.length; i++) {
    const [tx, ty] = applyAffine(result.m, a[i][0], a[i][1]);
    ss += (tx - b[i][0]) ** 2 + (ty - b[i][1]) ** 2;
  }
  return { ...result, n: a.length, rms: Math.sqrt(ss / a.length) };
}

/**
 * Greedy nearest-neighbour pairing (PsyFit's _compute_coloc_mask): each point of
 * A takes its closest unused point of B within `radius`.
 * @returns {{pairs: Array<{i:number, j:number, d:number}>, aMask: boolean[], bMask: boolean[]}}
 */
export function matchPoints(A, B, radius) {
  const aMask = new Array(A.length).fill(false), bMask = new Array(B.length).fill(false);
  const pairs = [];
  for (let i = 0; i < A.length; i++) {
    let best = -1, bd = Infinity;
    for (let j = 0; j < B.length; j++) {
      if (bMask[j]) continue;
      const d = Math.hypot(B[j][0] - A[i][0], B[j][1] - A[i][1]);
      if (d < bd) { bd = d; best = j; }
    }
    if (best >= 0 && bd <= radius) {
      aMask[i] = true; bMask[best] = true;
      pairs.push({ i, j: best, d: bd });
    }
  }
  return { pairs, aMask, bMask };
}

// ---------------------------------------------------------------------------
// Aperture (field stop) circle
// ---------------------------------------------------------------------------

/** Algebraic (Kasa) least-squares circle. */
export function fitCircle(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  const A = new Array(9).fill(0), b = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const x = xs[i] - mx, y = ys[i] - my;
    const r = [x, y, 1], rhs = -(x * x + y * y);
    for (let p = 0; p < 3; p++) {
      for (let q = 0; q < 3; q++) A[p * 3 + q] += r[p] * r[q];
      b[p] += r[p] * rhs;
    }
  }
  const sol = solve3(A, b);
  if (!sol) return null;
  const cx = -sol[0] / 2, cy = -sol[1] / 2;
  const val = cx * cx + cy * cy - sol[2];
  if (!(val > 0)) return null;
  return { cx: cx + mx, cy: cy + my, r: Math.sqrt(val) };
}

function otsuThreshold(values, min, max) {
  const NB = 256;
  const hist = new Float64Array(NB);
  const k = (NB - 1e-9) / (max - min);
  for (let i = 0; i < values.length; i++) hist[Math.floor((values[i] - min) * k)]++;
  const total = values.length;
  let sumAll = 0;
  for (let i = 0; i < NB; i++) sumAll += i * hist[i];
  let wB = 0, sumB = 0, best = -1, arg = 0;
  for (let i = 0; i < NB - 1; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; arg = i; }
  }
  return min + (arg + 0.5) * (max - min) / NB;
}

/**
 * Fit the circular illuminated aperture (field stop) in an image, after PsyFit's
 * fit_aperture_circle: blur, Otsu threshold, take the bright component covering
 * the image centre (else the largest), fill holes, then fit a circle to its
 * boundary arc with outlier rejection so a clipped edge does not bias it.
 *
 * The step between the dark surround and the illuminated disc is small next to
 * the particles inside it, so intensities are first clipped at `clipPercentile`
 * (the disc background sits near the median): otherwise Otsu separates
 * particles from background instead of disc from surround. The result is
 * rejected unless it is a plausible field stop for the image (radius, centre
 * and how well the region fills its circle).
 * @returns {null | {cx:number, cy:number, r:number, areaPx:number, fit:'circle'|'area', rms:number}}
 */
export function fitApertureCircle(img, w, h, opts = {}) {
  const sigma = opts.sigma ?? 5;
  const clipPct = opts.clipPercentile ?? 60;
  const size = Math.min(w, h);
  const minRadius = opts.minRadiusPx ?? 0.2 * size;
  const maxRadius = opts.maxRadiusPx ?? 0.75 * size;
  if (w < 16 || h < 16) return null;
  let src = img;
  if (clipPct < 100) {
    const sorted = Float32Array.from(img).sort();
    const cap = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * clipPct / 100))];
    src = new Float32Array(img.length);
    for (let i = 0; i < img.length; i++) src[i] = img[i] > cap ? cap : img[i];
  }
  const sm = gaussianFilter(src, w, h, sigma);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < sm.length; i++) { const v = sm[i]; if (v < min) min = v; if (v > max) max = v; }
  if (!(max > min)) return null;
  const thr = otsuThreshold(sm, min, max);
  const mask = new Uint8Array(w * h);
  let on = 0;
  for (let i = 0; i < sm.length; i++) if (sm[i] > thr) { mask[i] = 1; on++; }
  if (on < 0.02 * w * h) return null;

  // 4-connected components.
  const labels = new Int32Array(w * h);
  const sizes = [0];
  const stack = [];
  let nLabels = 0;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start]) continue;
    nLabels++;
    let size = 0;
    stack.push(start);
    labels[start] = nLabels;
    while (stack.length) {
      const p = stack.pop();
      size++;
      const x = p % w, y = (p / w) | 0;
      if (x > 0 && mask[p - 1] && !labels[p - 1]) { labels[p - 1] = nLabels; stack.push(p - 1); }
      if (x < w - 1 && mask[p + 1] && !labels[p + 1]) { labels[p + 1] = nLabels; stack.push(p + 1); }
      if (y > 0 && mask[p - w] && !labels[p - w]) { labels[p - w] = nLabels; stack.push(p - w); }
      if (y < h - 1 && mask[p + w] && !labels[p + w]) { labels[p + w] = nLabels; stack.push(p + w); }
    }
    sizes.push(size);
  }
  if (!nLabels) return null;
  let chosen = labels[((h / 2) | 0) * w + ((w / 2) | 0)];
  if (!chosen) {
    chosen = 1;
    for (let l = 2; l <= nLabels; l++) if (sizes[l] > sizes[chosen]) chosen = l;
  }
  // Fill holes: background reachable from the frame stays background.
  const disk = new Uint8Array(w * h);
  for (let i = 0; i < disk.length; i++) disk[i] = labels[i] === chosen ? 1 : 0;
  const outside = new Uint8Array(w * h);
  const push = (p) => { if (!disk[p] && !outside[p]) { outside[p] = 1; stack.push(p); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop();
    const x = p % w, y = (p / w) | 0;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w);
    if (y < h - 1) push(p + w);
  }
  let area = 0, sxm = 0, sym = 0;
  for (let i = 0; i < disk.length; i++) {
    if (!outside[i]) { disk[i] = 1; area++; sxm += i % w; sym += (i / w) | 0; }
  }
  const fallback = { cx: sxm / area, cy: sym / area, r: Math.sqrt(area / Math.PI), areaPx: area, fit: 'area', rms: NaN };
  const valid = (c) => {
    if (!c || !(c.r >= minRadius && c.r <= maxRadius)) return false;
    if (Math.abs(c.cx - w / 2) > 0.35 * w || Math.abs(c.cy - h / 2) > 0.35 * h) return false;
    // the bright region should fill most of the part of its circle that lies inside the image
    let inImage = 0, lit = 0;
    const r2 = c.r * c.r;
    for (let y = 0; y < h; y += 2) {
      for (let x = 0; x < w; x += 2) {
        if ((x - c.cx) ** 2 + (y - c.cy) ** 2 <= r2) { inImage++; if (disk[y * w + x]) lit++; }
      }
    }
    if (!(inImage > 0) || lit / inImage < 0.8) return false;
    return !(c.rms > 0.06 * c.r);
  };

  // Boundary pixels, excluding those on the image frame (a crop edge is not an arc).
  let xs = [], ys = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      if (!disk[p]) continue;
      if (!disk[p - 1] || !disk[p + 1] || !disk[p - w] || !disk[p + w]) { xs.push(x); ys.push(y); }
    }
  }
  if (xs.length < 8) return valid(fallback) ? fallback : null;
  let c = fitCircle(xs, ys);
  for (let it = 0; it < 8 && c; it++) {
    const res = xs.map((x, i) => Math.abs(Math.hypot(x - c.cx, ys[i] - c.cy) - c.r));
    const sorted = Float64Array.from(res).sort();
    const med = sorted[sorted.length >> 1];
    const dev = Float64Array.from(res, (r) => Math.abs(r - med)).sort();
    const mad = dev[dev.length >> 1] + 1e-9;
    const limit = med + 2.0 * 1.4826 * mad;
    const kx = [], ky = [];
    for (let i = 0; i < xs.length; i++) if (res[i] < limit) { kx.push(xs[i]); ky.push(ys[i]); }
    if (kx.length < 8 || kx.length === xs.length) break;
    xs = kx; ys = ky;
    c = fitCircle(xs, ys);
  }
  if (!c) return valid(fallback) ? fallback : null;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) ss += (Math.hypot(xs[i] - c.cx, ys[i] - c.cy) - c.r) ** 2;
  const out = { cx: c.cx, cy: c.cy, r: c.r, areaPx: area, fit: 'circle', rms: Math.sqrt(ss / xs.length) };
  return valid(out) ? out : null;
}

/**
 * Fit the aperture in several images of one acquisition (one per channel) with a
 * shared clipping level, so soft disc edges bias every channel the same way.
 * Tries a few levels and keeps the one that finds the most apertures.
 * @param {Array<{data: Float32Array, w: number, h: number}>} images
 * @returns {{circles: Array<object|null>, found: number, clipPercentile: number}}
 */
export function fitApertures(images, opts = {}) {
  let best = { circles: images.map(() => null), found: 0, clipPercentile: 60, score: Infinity };
  for (const clip of opts.levels ?? [60, 50, 70, 80]) {
    const circles = images.map((im) => fitApertureCircle(im.data, im.w, im.h, { ...opts, clipPercentile: clip }));
    const ok = circles.filter(Boolean);
    const score = ok.length ? ok.reduce((a, c) => a + (c.rms || 0) / c.r, 0) / ok.length : Infinity;
    if (ok.length > best.found || (ok.length === best.found && score < best.score)) {
      best = { circles, found: ok.length, clipPercentile: clip, score };
    }
    if (ok.length === images.length) break;
  }
  return best;
}
