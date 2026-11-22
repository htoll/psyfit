// Image filters, peak finding and PSF fitting.
//
// Detection and the single-PSF fit follow PsyFit's `integrate_sif` /
// `fit_psf_brightness` step for step (Gaussian blur, skimage-style
// peak_local_max, a bounded 2-D Gaussian least-squares fit on the 20x20
// interpolated window, brightness = window sum - N * fitted offset), so numbers
// agree with the desktop tools. Images are Float32Array, row-major, row 0 first;
// "x" is the column and "y" the row, as in PsyFit's x_pix / y_pix.

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

function gaussianWeights(sigma, order = 0, truncate = 4.0) {
  const radius = Math.floor(truncate * sigma + 0.5);
  const w = new Float64Array(radius + 1);     // w[k] for offset +-k (even kernels)
  let sum = 0;
  for (let k = -radius; k <= radius; k++) sum += Math.exp(-0.5 * k * k / (sigma * sigma));
  for (let k = 0; k <= radius; k++) {
    const phi = Math.exp(-0.5 * k * k / (sigma * sigma)) / sum;
    w[k] = order === 2 ? phi * (k * k / (sigma ** 4) - 1 / (sigma * sigma)) : phi;
  }
  return { radius, w };
}

function reflectIndex(i, n) {
  if (i >= 0 && i < n) return i;
  const period = 2 * n;
  let j = ((i % period) + period) % period;
  if (j >= n) j = period - 1 - j;
  return j;
}

/** Symmetric 1-D correlation along one axis with scipy's "reflect" boundary; float32 output. */
function correlateAxis(src, dst, w, h, kernel, vertical) {
  const { radius, w: kw } = kernel;
  const n = vertical ? h : w;
  const lines = vertical ? w : h;
  const line = new Float64Array(n + 2 * radius);
  for (let l = 0; l < lines; l++) {
    const base = vertical ? l : l * w;
    const stride = vertical ? w : 1;
    for (let i = -radius; i < n + radius; i++) line[i + radius] = src[base + reflectIndex(i, n) * stride];
    for (let i = 0; i < n; i++) {
      const c = i + radius;
      let acc = line[c] * kw[0];
      for (let k = radius; k >= 1; k--) acc += (line[c - k] + line[c + k]) * kw[k];
      dst[base + i * stride] = acc;
    }
  }
}

/** scipy.ndimage.gaussian_filter(img, sigma) for a float32 image. */
export function gaussianFilter(src, w, h, sigma, orders = [0, 0]) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  correlateAxis(src, tmp, w, h, gaussianWeights(sigma, orders[0]), true);
  correlateAxis(tmp, out, w, h, gaussianWeights(sigma, orders[1]), false);
  return out;
}

/** Square maximum filter of half-width r, edges replicated ("nearest"). */
export function maximumFilter(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const a = Math.max(0, x - r), b = Math.min(w - 1, x + r);
      let m = src[row + a];
      for (let i = a + 1; i <= b; i++) { const v = src[row + i]; if (v > m) m = v; }
      tmp[row + x] = m;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const a = Math.max(0, y - r), b = Math.min(h - 1, y + r);
      let m = tmp[a * w + x];
      for (let i = a + 1; i <= b; i++) { const v = tmp[i * w + x]; if (v > m) m = v; }
      out[y * w + x] = m;
    }
  }
  return out;
}

/** Mean and population standard deviation. */
export function meanStd(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  const mean = s / a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - mean; v += d * d; }
  return { mean, std: Math.sqrt(v / a.length) };
}

// ---------------------------------------------------------------------------
// Peak finding (skimage.feature.peak_local_max semantics)
// ---------------------------------------------------------------------------

/**
 * @returns {Array<[number, number]>} [row, col] sorted brightest first
 */
export function peakLocalMax(img, w, h, opts = {}) {
  const minDistance = Math.max(1, Math.floor(opts.minDistance ?? 1));
  const numPeaks = opts.numPeaks ?? Infinity;
  const border = opts.excludeBorder === false ? 0 : minDistance;
  let threshold = opts.thresholdAbs;
  if (threshold === undefined || threshold === null) {
    threshold = Infinity;
    for (let i = 0; i < img.length; i++) if (img[i] < threshold) threshold = img[i];
  }
  if (w * h <= 1) return [];
  const mx = maximumFilter(img, w, h, minDistance);
  let trivial = true;
  for (let i = 0; i < img.length; i++) if (img[i] !== mx[i]) { trivial = false; break; }
  if (trivial) return [];

  const idx = [];
  for (let y = border; y < h - border; y++) {
    const row = y * w;
    for (let x = border; x < w - border; x++) {
      const v = img[row + x];
      if (v === mx[row + x] && v > threshold) idx.push(row + x);
    }
  }
  // Stable sort, brightest first (ties keep row-major order).
  idx.sort((a, b) => (img[b] - img[a]) || (a - b));

  // Greedy spacing: drop a candidate within Chebyshev distance < minDistance of a kept one.
  const out = [];
  const taken = minDistance > 1 ? new Uint8Array(w * h) : null;
  const reach = minDistance - 1;
  for (let i = 0; i < idx.length && out.length < numPeaks; i++) {
    const p = idx[i];
    const y = (p / w) | 0, x = p - y * w;
    if (taken) {
      let clash = false;
      const y0 = Math.max(0, y - reach), y1 = Math.min(h - 1, y + reach);
      const x0 = Math.max(0, x - reach), x1 = Math.min(w - 1, x + reach);
      for (let yy = y0; yy <= y1 && !clash; yy++) {
        for (let xx = x0; xx <= x1; xx++) if (taken[yy * w + xx]) { clash = true; break; }
      }
      if (clash) continue;
      taken[p] = 1;
    }
    out.push([y, x]);
  }
  return out;
}

/**
 * PsyFit's detection step: blur with sigma 1, threshold at mean + k * std of the
 * blurred image, then local maxima ("UCNP") or Laplacian-of-Gaussian blobs ("dye").
 * @returns {{coords: Array<[number, number]>, smoothed: Float32Array, thresholdAbs: number}}
 */
export function detectPeaks(img, w, h, opts = {}) {
  const threshold = opts.threshold ?? 10;
  const minDistance = opts.minDistance ?? 5;
  const smoothed = gaussianFilter(img, w, h, 1);
  const { mean, std } = meanStd(smoothed);
  const thresholdAbs = mean + threshold * std;
  let coords;
  if (opts.signal === 'dye') {
    coords = blobLog(smoothed, w, h, { minSigma: 1, maxSigma: 3, numSigma: 5, threshold: 5 * threshold })
      .map((b) => [b[0], b[1]]);
  } else {
    coords = peakLocalMax(smoothed, w, h, { minDistance, thresholdAbs });
  }
  return { coords, smoothed, thresholdAbs };
}

// ---------------------------------------------------------------------------
// Laplacian-of-Gaussian blobs (skimage.feature.blob_log)
// ---------------------------------------------------------------------------

function diskOverlap(d, r1, r2) {
  const clip = (v) => Math.max(-1, Math.min(1, v));
  const acos1 = Math.acos(clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1)));
  const acos2 = Math.acos(clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2)));
  const a = -d + r2 + r1, b = d - r2 + r1, c = d + r2 - r1, e = d + r2 + r1;
  const area = r1 * r1 * acos1 + r2 * r2 * acos2 - 0.5 * Math.sqrt(Math.abs(a * b * c * e));
  return area / (Math.PI * Math.min(r1, r2) ** 2);
}

function blobOverlap(b1, b2) {
  const root = Math.SQRT2;
  if (b1[2] === 0 && b2[2] === 0) return 0;
  let r1, r2, maxSigma;
  if (b1[2] > b2[2]) { maxSigma = b1[2]; r1 = 1; r2 = b2[2] / b1[2]; }
  else { maxSigma = b2[2]; r2 = 1; r1 = b1[2] / b2[2]; }
  const s = maxSigma * root;
  const d = Math.hypot((b2[0] - b1[0]) / s, (b2[1] - b1[1]) / s);
  if (d > r1 + r2) return 0;
  if (d <= Math.abs(r1 - r2)) return 1;
  return diskOverlap(d, r1, r2);
}

/** @returns {Array<[number, number, number]>} [row, col, sigma], strongest first */
export function blobLog(img, w, h, opts = {}) {
  const { minSigma = 1, maxSigma = 3, numSigma = 5, threshold = 0.2, overlap = 0.5 } = opts;
  const sigmas = [];
  for (let i = 0; i < numSigma; i++) sigmas.push(numSigma === 1 ? minSigma : minSigma + (maxSigma - minSigma) * i / (numSigma - 1));
  const n = w * h;
  const cube = sigmas.map((s) => {
    const a = gaussianFilter(img, w, h, s, [2, 0]);
    const b = gaussianFilter(img, w, h, s, [0, 2]);
    const out = new Float32Array(n);
    const k = -s * s;
    for (let i = 0; i < n; i++) out[i] = (a[i] + b[i]) * k;
    return out;
  });
  const S = sigmas.length;
  const peaks = [];
  for (let s = 0; s < S; s++) {
    const cur = cube[s];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = cur[y * w + x];
        if (!(v > threshold)) continue;
        let isMax = true;
        for (let ds = -1; ds <= 1 && isMax; ds++) {
          const layer = cube[Math.max(0, Math.min(S - 1, s + ds))];
          for (let dy = -1; dy <= 1 && isMax; dy++) {
            const yy = Math.max(0, Math.min(h - 1, y + dy));
            for (let dx = -1; dx <= 1; dx++) {
              const xx = Math.max(0, Math.min(w - 1, x + dx));
              if (layer[yy * w + xx] > v) { isMax = false; break; }
            }
          }
        }
        if (isMax) peaks.push([y, x, sigmas[s], v]);
      }
    }
  }
  peaks.sort((a, b) => b[3] - a[3]);
  if (peaks.length > 1) {
    let sigmaMax = 0;
    for (const p of peaks) if (p[2] > sigmaMax) sigmaMax = p[2];
    const reach = 2 * sigmaMax * Math.SQRT2;
    const cell = Math.max(1, Math.ceil(reach));
    const gw = Math.ceil(w / cell) + 1;
    const grid = new Map();
    peaks.forEach((p, i) => {
      const key = Math.floor(p[0] / cell) * gw + Math.floor(p[1] / cell);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    });
    for (let i = 0; i < peaks.length; i++) {
      const p = peaks[i];
      const gy = Math.floor(p[0] / cell), gx = Math.floor(p[1] / cell);
      for (let yy = gy - 1; yy <= gy + 1; yy++) {
        for (let xx = gx - 1; xx <= gx + 1; xx++) {
          const bucket = grid.get(yy * gw + xx);
          if (!bucket) continue;
          for (const j of bucket) {
            if (j <= i) continue;
            const q = peaks[j];
            if (Math.hypot(p[0] - q[0], p[1] - q[1]) > reach) continue;
            if (blobOverlap(p, q) > overlap) {
              if (p[2] > q[2]) q[2] = 0; else p[2] = 0;
            }
          }
        }
      }
    }
  }
  return peaks.filter((p) => p[2] > 0).map((p) => [p[0], p[1], p[2]]);
}

// ---------------------------------------------------------------------------
// Bounded nonlinear least squares: a port of SciPy's Trust Region Reflective
// solver (scipy.optimize.least_squares, method="trf", exact trust-region
// subproblem), which is what PsyFit's fits run on. Working from the normal
// matrix J'J is enough here because every quantity TRF needs from J is a
// quadratic form.
// ---------------------------------------------------------------------------

const EPS = 2.220446049250313e-16;

function jacobiEigen(B, n) {
  // Symmetric eigen-decomposition (cyclic Jacobi). Returns eigenvalues sorted
  // descending and the matching eigenvectors as columns of V (row-major).
  const A = Float64Array.from(B);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0, diag = 0;
    for (let i = 0; i < n; i++) {
      diag += A[i * n + i] * A[i * n + i];
      for (let j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
    }
    if (off <= 1e-30 * (diag + 1e-300)) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (apq === 0) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j * n + j] - A[i * n + i]);
  const evals = new Float64Array(n), Vs = new Float64Array(n * n);
  order.forEach((src, dst) => {
    evals[dst] = A[src * n + src];
    for (let k = 0; k < n; k++) Vs[k * n + dst] = V[k * n + src];
  });
  return { evals, V: Vs };
}

const vnorm = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
const vdot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function quadForm(B, n, a, b) {
  let s = 0;
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let j = 0; j < n; j++) r += B[i * n + j] * b[j];
    s += a[i] * r;
  }
  return s;
}

function makeStrictlyFeasible(x, lb, ub, rstep) {
  const out = Float64Array.from(x);
  for (let i = 0; i < x.length; i++) {
    let active = 0;
    if (rstep === 0) {
      if (x[i] <= lb[i]) active = -1;
      if (x[i] >= ub[i]) active = 1;
    } else {
      const lo = x[i] - lb[i], hi = ub[i] - x[i];
      if (Number.isFinite(lb[i]) && lo <= Math.min(hi, rstep * Math.max(1, Math.abs(lb[i])))) active = -1;
      if (Number.isFinite(ub[i]) && hi <= Math.min(lo, rstep * Math.max(1, Math.abs(ub[i])))) active = 1;
    }
    if (active === -1) out[i] = rstep === 0 ? nextAfter(lb[i], 1) : lb[i] + rstep * Math.max(1, Math.abs(lb[i]));
    else if (active === 1) out[i] = rstep === 0 ? nextAfter(ub[i], -1) : ub[i] - rstep * Math.max(1, Math.abs(ub[i]));
    if (out[i] < lb[i] || out[i] > ub[i]) out[i] = 0.5 * (lb[i] + ub[i]);
  }
  return out;
}

const F64 = new Float64Array(1), U64 = new BigUint64Array(F64.buffer);
function nextAfter(x, dir) {
  if (!Number.isFinite(x)) return x;
  if (x === 0) return dir > 0 ? 5e-324 : -5e-324;
  F64[0] = x;
  if ((x > 0) === (dir > 0)) U64[0] += 1n; else U64[0] -= 1n;
  return F64[0];
}

function stepSizeToBound(x, s, lb, ub) {
  const n = x.length;
  const steps = new Float64Array(n).fill(Infinity);
  let min = Infinity;
  for (let i = 0; i < n; i++) {
    if (s[i] !== 0) steps[i] = Math.max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
    if (steps[i] < min) min = steps[i];
  }
  const hits = new Int8Array(n);
  for (let i = 0; i < n; i++) if (steps[i] === min) hits[i] = Math.sign(s[i]);
  return { min, hits };
}

function minimizeQuadratic1d(a, b, lo, hi, c = 0) {
  const ts = [lo, hi];
  if (a !== 0) {
    const ext = -0.5 * b / a;
    if (lo < ext && ext < hi) ts.push(ext);
  }
  let bestT = ts[0], bestY = Infinity;
  for (const t of ts) {
    const y = t * (a * t + b) + c;
    if (y < bestY) { bestY = y; bestT = t; }
  }
  return [bestT, bestY];
}

function solveTrustRegion(n, m, suf, s, V, Delta, initialAlpha) {
  const p = new Float64Array(n);
  const stepFor = (alpha) => {
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += V[i * n + k] * (suf[k] / (s[k] * s[k] + alpha));
      p[i] = -acc;
    }
  };
  const phiAnd = (alpha) => {
    let pn = 0, dsum = 0;
    for (let k = 0; k < n; k++) {
      const den = s[k] * s[k] + alpha;
      pn += (suf[k] / den) ** 2;
      dsum += suf[k] * suf[k] / (den * den * den);
    }
    pn = Math.sqrt(pn);
    return [pn - Delta, -dsum / pn];
  };
  const fullRank = m >= n && s[n - 1] > EPS * m * s[0];
  if (fullRank) {
    stepFor(0);
    if (vnorm(p) <= Delta) return { p, alpha: 0 };
  }
  let alphaUpper = vnorm(suf) / Delta;
  let alphaLower = 0;
  if (fullRank) {
    const [phi, phiPrime] = phiAnd(0);
    alphaLower = -phi / phiPrime;
  }
  let alpha;
  if (initialAlpha === null || (!fullRank && initialAlpha === 0)) {
    alpha = Math.max(0.001 * alphaUpper, Math.sqrt(alphaLower * alphaUpper));
  } else alpha = initialAlpha;
  for (let it = 0; it < 10; it++) {
    if (alpha < alphaLower || alpha > alphaUpper) alpha = Math.max(0.001 * alphaUpper, Math.sqrt(alphaLower * alphaUpper));
    const [phi, phiPrime] = phiAnd(alpha);
    if (phi < 0) alphaUpper = alpha;
    const ratio = phi / phiPrime;
    alphaLower = Math.max(alphaLower, alpha - ratio);
    alpha -= (phi + Delta) * ratio / Delta;
    if (Math.abs(phi) < 0.01 * Delta) break;
  }
  stepFor(alpha);
  const pn = vnorm(p);
  if (pn > 0) for (let i = 0; i < n; i++) p[i] *= Delta / pn;
  return { p, alpha };
}

/**
 * Minimise 0.5 * sum(r^2) subject to lb <= x <= ub.
 * @param {(x: Float64Array, H: Float64Array|null, g: Float64Array|null) => number} fun
 *   returns sum(r^2); when H/g are given (zero-filled) it also accumulates
 *   J'J into H (n*n, row-major, full) and J'r into g.
 * @param {{m?: number, xScale?: 'jac', maxNfev?: number, ftol?: number, xtol?: number, gtol?: number}} [opts]
 */
export function trfLeastSquares(fun, x0, lb, ub, opts = {}) {
  const n = x0.length;
  const ftol = opts.ftol ?? 1e-8, xtol = opts.xtol ?? 1e-8, gtol = opts.gtol ?? 1e-8;
  const maxNfev = opts.maxNfev ?? 100 * n;
  const m = opts.m ?? Infinity;
  const jacScale = opts.xScale === 'jac';
  let x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.min(ub[i], Math.max(lb[i], x0[i]));
  x = makeStrictlyFeasible(x, lb, ub, 1e-10);
  const H = new Float64Array(n * n), g = new Float64Array(n);
  let cost = 0.5 * fun(x, H, g);
  let nfev = 1;
  const scale = new Float64Array(n).fill(1), scaleInv = new Float64Array(n).fill(1);
  const updateScale = (first) => {
    for (let i = 0; i < n; i++) {
      let si = Math.sqrt(H[i * n + i]);
      if (first) { if (si === 0) si = 1; } else si = Math.max(si, scaleInv[i]);
      scaleInv[i] = si; scale[i] = 1 / si;
    }
  };
  if (jacScale) updateScale(true);

  const v = new Float64Array(n), dv = new Float64Array(n);
  const clScaling = () => {
    for (let i = 0; i < n; i++) {
      v[i] = 1; dv[i] = 0;
      if (g[i] < 0 && Number.isFinite(ub[i])) { v[i] = ub[i] - x[i]; dv[i] = -1; }
      if (g[i] > 0 && Number.isFinite(lb[i])) { v[i] = x[i] - lb[i]; dv[i] = 1; }
    }
  };
  clScaling();
  for (let i = 0; i < n; i++) if (dv[i] !== 0) v[i] *= scaleInv[i];
  let Delta = 0;
  for (let i = 0; i < n; i++) Delta += (x[i] * scaleInv[i] / Math.sqrt(v[i])) ** 2;
  Delta = Math.sqrt(Delta);
  if (!(Delta > 0) || !Number.isFinite(Delta)) Delta = 1;

  const d = new Float64Array(n), diagH = new Float64Array(n), gH = new Float64Array(n);
  const B = new Float64Array(n * n), s = new Float64Array(n), suf = new Float64Array(n);
  const tmp = new Float64Array(n);
  let alpha = 0, status = null, iter = 0;
  if (!Number.isFinite(cost)) return { x, cost, nfev, status: -1, iter };

  for (;;) {
    clScaling();
    let gNorm = 0;
    for (let i = 0; i < n; i++) gNorm = Math.max(gNorm, Math.abs(g[i] * v[i]));
    if (gNorm < gtol) status = 1;
    if (status !== null || nfev >= maxNfev) break;
    for (let i = 0; i < n; i++) {
      if (dv[i] !== 0) v[i] *= scaleInv[i];
      d[i] = Math.sqrt(v[i]) * scale[i];
      diagH[i] = g[i] * dv[i] * scale[i];
      gH[i] = d[i] * g[i];
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) B[i * n + j] = d[i] * H[i * n + j] * d[j];
      B[i * n + i] += diagH[i];
    }
    const eig = jacobiEigen(B, n);
    const V = eig.V;
    for (let k = 0; k < n; k++) {
      s[k] = Math.sqrt(Math.max(eig.evals[k], 0));
      let acc = 0;
      for (let i = 0; i < n; i++) acc += V[i * n + k] * gH[i];
      suf[k] = acc;
    }
    const theta = Math.max(0.995, 1 - gNorm);
    let actual = -1;
    let xNew = x, costNew = cost;
    while (actual <= 0 && nfev < maxNfev) {
      const tr = solveTrustRegion(n, m, suf, s, V, Delta, alpha);
      alpha = tr.alpha;
      const pH = tr.p;
      const p = new Float64Array(n);
      for (let i = 0; i < n; i++) p[i] = d[i] * pH[i];

      // ---- select_step: plain, reflected or constrained-Cauchy step ----
      let step, stepH, predicted;
      let inside = true;
      for (let i = 0; i < n; i++) { const t = x[i] + p[i]; if (!(t >= lb[i] && t <= ub[i])) { inside = false; break; } }
      if (inside) {
        step = p; stepH = pH;
        predicted = -(0.5 * quadForm(B, n, pH, pH) + vdot(pH, gH));
      } else {
        const hit = stepSizeToBound(x, p, lb, ub);
        const pStride = hit.min;
        const rH = Float64Array.from(pH);
        for (let i = 0; i < n; i++) if (hit.hits[i] !== 0) rH[i] = -rH[i];
        const r = new Float64Array(n);
        for (let i = 0; i < n; i++) r[i] = d[i] * rH[i];
        for (let i = 0; i < n; i++) { p[i] *= pStride; pH[i] *= pStride; tmp[i] = x[i] + p[i]; }
        // intersect the reflected ray with the trust region
        let toTr = 0;
        {
          const a = vdot(rH, rH), b = vdot(pH, rH), c = vdot(pH, pH) - Delta * Delta;
          const disc = Math.sqrt(Math.max(b * b - a * c, 0));
          const q = -(b + (b < 0 || Object.is(b, -0) ? -disc : disc));
          const t1 = q / a, t2 = q !== 0 ? c / q : 0;
          toTr = Math.max(t1, t2);
        }
        const toBound = stepSizeToBound(tmp, r, lb, ub).min;
        let rStride = Math.min(toBound, toTr), rLo, rHi;
        if (rStride > 0) {
          rLo = (1 - theta) * pStride / rStride;
          rHi = rStride === toBound ? theta * toBound : toTr;
        } else { rLo = 0; rHi = -1; }
        let rValue = Infinity;
        if (rLo <= rHi) {
          const a = 0.5 * quadForm(B, n, rH, rH);
          const b = vdot(gH, rH) + quadForm(B, n, pH, rH);
          const c = 0.5 * quadForm(B, n, pH, pH) + vdot(gH, pH);
          const [t, val] = minimizeQuadratic1d(a, b, rLo, rHi, c);
          rValue = val;
          for (let i = 0; i < n; i++) { rH[i] = rH[i] * t + pH[i]; r[i] = rH[i] * d[i]; }
        }
        for (let i = 0; i < n; i++) { p[i] *= theta; pH[i] *= theta; }
        const pValue = 0.5 * quadForm(B, n, pH, pH) + vdot(pH, gH);
        const agH = new Float64Array(n), ag = new Float64Array(n);
        for (let i = 0; i < n; i++) { agH[i] = -gH[i]; ag[i] = d[i] * agH[i]; }
        const toTrAg = Delta / vnorm(agH);
        const toBoundAg = stepSizeToBound(x, ag, lb, ub).min;
        const agMax = toBoundAg < toTrAg ? theta * toBoundAg : toTrAg;
        const [agStride, agValue] = minimizeQuadratic1d(0.5 * quadForm(B, n, agH, agH), vdot(gH, agH), 0, agMax);
        for (let i = 0; i < n; i++) { agH[i] *= agStride; ag[i] *= agStride; }
        if (pValue < rValue && pValue < agValue) { step = p; stepH = pH; predicted = -pValue; }
        else if (rValue < pValue && rValue < agValue) { step = r; stepH = rH; predicted = -rValue; }
        else { step = ag; stepH = agH; predicted = -agValue; }
      }

      for (let i = 0; i < n; i++) tmp[i] = x[i] + step[i];
      xNew = makeStrictlyFeasible(tmp, lb, ub, 0);
      const ssNew = fun(xNew, null, null);
      nfev++;
      const stepHNorm = vnorm(stepH);
      if (!Number.isFinite(ssNew)) { Delta = 0.25 * stepHNorm; continue; }
      costNew = 0.5 * ssNew;
      actual = cost - costNew;
      let ratio;
      if (predicted > 0) ratio = actual / predicted;
      else if (predicted === 0 && actual === 0) ratio = 1;
      else ratio = 0;
      let DeltaNew = Delta;
      if (ratio < 0.25) DeltaNew = 0.25 * stepHNorm;
      else if (ratio > 0.75 && stepHNorm > 0.95 * Delta) DeltaNew = Delta * 2;
      const stepNorm = vnorm(step), xNorm = vnorm(x);
      const fOk = actual < ftol * cost && ratio > 0.25;
      const xOk = stepNorm < xtol * (xtol + xNorm);
      if (fOk && xOk) status = 4; else if (fOk) status = 2; else if (xOk) status = 3;
      if (status !== null) break;
      alpha *= Delta / DeltaNew;
      Delta = DeltaNew;
    }
    if (actual > 0) {
      x = xNew;
      H.fill(0); g.fill(0);
      fun(x, H, g);
      cost = costNew;
      if (jacScale) updateScale(false);
    }
    iter++;
  }
  return { x, cost, nfev, status: status === null ? 0 : status, iter };
}

// ---------------------------------------------------------------------------
// Single-PSF fit (PsyFit-compatible)
// ---------------------------------------------------------------------------

function extractWindow(img, w, h, cx, cy, r) {
  const x0 = Math.trunc(Math.max(cx - r, 0));
  const x1 = Math.trunc(Math.min(cx + r + 1, w));
  const y0 = Math.trunc(Math.max(cy - r, 0));
  const y1 = Math.trunc(Math.min(cy + r + 1, h));
  const sw = Math.max(0, x1 - x0), sh = Math.max(0, y1 - y0);
  const sub = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    const src = (y0 + y) * w + x0;
    for (let x = 0; x < sw; x++) sub[y * sw + x] = img[src + x];
  }
  return { sub, sw, sh, x0, y0 };
}

/** scipy.ndimage.zoom(order=1) to `outH` rows (and the matching column count). */
function bilinearResample(sub, sw, sh, outW, outH) {
  const out = new Float64Array(outW * outH);
  const fy = outH > 1 ? (sh - 1) / (outH - 1) : 0;
  const fx = outW > 1 ? (sw - 1) / (outW - 1) : 0;
  for (let j = 0; j < outH; j++) {
    const y = j * fy;
    let yi = Math.floor(y);
    if (yi > sh - 2) yi = Math.max(0, sh - 2);
    const ty = sh > 1 ? y - yi : 0;
    const r0 = yi * sw, r1 = Math.min(sh - 1, yi + 1) * sw;
    for (let i = 0; i < outW; i++) {
      const x = i * fx;
      let xi = Math.floor(x);
      if (xi > sw - 2) xi = Math.max(0, sw - 2);
      const tx = sw > 1 ? x - xi : 0;
      const xj = Math.min(sw - 1, xi + 1);
      const top = sub[r0 + xi] * (1 - tx) + sub[r0 + xj] * tx;
      const bot = sub[r1 + xi] * (1 - tx) + sub[r1 + xj] * tx;
      out[j * outW + i] = top * (1 - ty) + bot * ty;
    }
  }
  return out;
}

export const FIT_DEFAULTS = {
  pixUm: 0.1,          // um per pixel
  radiusUm: 0.5,       // half-width of the fit window
  sigmaUbUm: 0.5,      // upper bound on sigma during the fit
  sigThresholdUm: 0.3, // fits wider than this are flagged/rejected
  refine: true,        // snap to the brightest blurred pixel in the window first
  interp: 20,          // PsyFit fits a 20x20 bilinear resampling of the window; 0 = raw pixels
  freeOffset: false,   // PsyFit caps the background at 1.2 x the window minimum
  traceBounds: false,  // amplitude bounds of PsyFit's per-frame fit_psf_brightness
};

/**
 * Fit one 2-D Gaussian PSF near (cx, cy) (column, row).
 * Returns null when no usable window/peak exists. All lengths are in pixels.
 */
export function fitPsf(img, w, h, cx, cy, options = {}) {
  const o = { ...FIT_DEFAULTS, ...options };
  const pix = o.pixUm;
  const r = Math.max(1, Math.trunc(o.radiusUm / pix));
  let px = Math.round(cx), py = Math.round(cy);
  if (!(px >= 0 && px < w && py >= 0 && py < h)) return null;

  if (o.refine) {
    const first = extractWindow(img, w, h, px, py, r);
    if (first.sw < 3 || first.sh < 3) return null;
    const blurred = gaussianFilter(first.sub, first.sw, first.sh, 1);
    const peak = peakLocalMax(blurred, first.sw, first.sh, { minDistance: 1, numPeaks: 1 });
    if (!peak.length) return null;
    px = first.x0 + peak[0][1];
    py = first.y0 + peak[0][0];
  }

  const { sub, sw, sh, x0, y0 } = extractWindow(img, w, h, px, py, r);
  if (sw < 2 || sh < 2) return null;
  let vmax = -Infinity, vmin = Infinity, sum = 0;
  for (let i = 0; i < sub.length; i++) { const v = sub[i]; if (v > vmax) vmax = v; if (v < vmin) vmin = v; sum += v; }
  if (!Number.isFinite(vmax) || !Number.isFinite(vmin)) return null;

  // Samples the model is fitted to (PsyFit resamples the window to 20 rows).
  let zs, nx, ny, stepX, stepY;
  if (o.interp > 0) {
    ny = o.interp;
    nx = Math.max(2, Math.round(sw * (o.interp / sh)));
    zs = bilinearResample(sub, sw, sh, nx, ny);
    stepX = (sw - 1) / (nx - 1);
    stepY = (sh - 1) / (ny - 1);
  } else {
    ny = sh; nx = sw; zs = sub; stepX = 1; stepY = 1;
  }
  const xs = new Float64Array(nx), ys = new Float64Array(ny);
  for (let i = 0; i < nx; i++) xs[i] = (x0 + i * stepX) * pix;
  for (let j = 0; j < ny; j++) ys[j] = (y0 + j * stepY) * pix;

  // Parameters, guesses and bounds exactly as PsyFit: A, x0, sx, y0, sy, offset
  // with positions and widths in micrometres.
  const ampGuess = vmax, offGuess = vmin;
  const xg = px * pix, yg = py * pix;
  const sigUb = o.sigmaUbUm;
  const sig0 = Math.min(0.3, sigUb * 0.999);
  // integrate_sif bounds the amplitude to [1, 2*max]; fit_psf_brightness (per-frame
  // traces, where a spot may be dark) uses [0, max(2*max, max+1, 2)].
  let ampLb = 1, ampUb = 2 * ampGuess;
  if (o.traceBounds || !(ampUb > ampLb) || !(ampGuess >= ampLb)) { ampLb = 0; ampUb = Math.max(2 * ampGuess, ampGuess + 1, 2); }
  const offUb = o.freeOffset ? Math.max(vmax, 1e-6) : Math.max(offGuess * 1.2, 1e-6);
  const offLb = o.freeOffset ? Math.min(0, vmin) : 0;
  const p0 = [ampGuess, xg, sig0, yg, sig0, offGuess];
  const lb = [ampLb, xg - 1, 0, yg - 1, 0, offLb];
  const ub = [ampUb, xg + 1, sigUb, yg + 1, sigUb, offUb];

  const model = (p, H, g) => {
    const A = p[0], mx = p[1], sx = p[2], my = p[3], sy = p[4], off = p[5];
    const isx2 = 1 / (sx * sx), isy2 = 1 / (sy * sy);
    let cost = 0;
    for (let j = 0; j < ny; j++) {
      const dy = ys[j] - my;
      const ey = Math.exp(-0.5 * dy * dy * isy2);
      for (let i = 0; i < nx; i++) {
        const dx = xs[i] - mx;
        const e = ey * Math.exp(-0.5 * dx * dx * isx2);
        const res = A * e + off - zs[j * nx + i];
        cost += res * res;
        if (H) {
          const ge = A * e;
          const j0 = e, j1 = ge * dx * isx2, j2 = ge * dx * dx * isx2 / sx;
          const j3 = ge * dy * isy2, j4 = ge * dy * dy * isy2 / sy;
          g[0] += j0 * res; g[1] += j1 * res; g[2] += j2 * res; g[3] += j3 * res; g[4] += j4 * res; g[5] += res;
          H[0] += j0 * j0; H[1] += j0 * j1; H[2] += j0 * j2; H[3] += j0 * j3; H[4] += j0 * j4; H[5] += j0;
          H[7] += j1 * j1; H[8] += j1 * j2; H[9] += j1 * j3; H[10] += j1 * j4; H[11] += j1;
          H[14] += j2 * j2; H[15] += j2 * j3; H[16] += j2 * j4; H[17] += j2;
          H[21] += j3 * j3; H[22] += j3 * j4; H[23] += j3;
          H[28] += j4 * j4; H[29] += j4;
          H[35] += 1;
        }
      }
    }
    if (H) {
      for (let a = 1; a < 6; a++) for (let b = 0; b < a; b++) H[a * 6 + b] = H[b * 6 + a];
    }
    return cost;
  };

  const nSamples = nx * ny;
  const fit = trfLeastSquares(model, p0, lb, ub, { m: nSamples });
  const A = fit.x[0], off = fit.x[5];
  const fx = fit.x[1] / pix, fy = fit.x[3] / pix;
  const sx = fit.x[2] / pix, sy = fit.x[4] / pix;
  const ss = 2 * fit.cost;
  let zmean = 0;
  for (let i = 0; i < nSamples; i++) zmean += zs[i];
  zmean /= nSamples;
  let sst = 0;
  for (let i = 0; i < nSamples; i++) sst += (zs[i] - zmean) ** 2;
  // Noise from the raw window (the resampled grid is smoother than the pixels).
  let rss = 0;
  for (let j = 0; j < sh; j++) {
    const dy = (y0 + j) - fy;
    for (let i = 0; i < sw; i++) {
      const dx = (x0 + i) - fx;
      const mdl = A * Math.exp(-0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))) + off;
      rss += (sub[j * sw + i] - mdl) ** 2;
    }
  }
  const noise = Math.sqrt(rss / Math.max(1, sub.length - 6));
  const brightnessFit = 2 * Math.PI * A * sx * sy;
  const brightness = sum - sub.length * off;
  const sigmaOk = sx * pix <= o.sigThresholdUm + 1e-3 && sy * pix <= o.sigThresholdUm + 1e-3;
  const brightnessOk = brightness >= 1 && brightness <= 1e9;
  return {
    x: fx, y: fy,                 // fitted centre (px, sub-pixel)
    peakX: px, peakY: py,         // refined peak pixel (PsyFit's x_pix / y_pix)
    amp: A, sx, sy, offset: off,
    brightness,                   // PsyFit brightness_integrated (pps)
    brightnessFit,                // 2*pi*A*sx*sy (pps)
    r2: sst > 0 ? 1 - ss / sst : 0,
    noise,
    snr: noise > 0 ? A / noise : Infinity,
    window: sub.length, winX: x0, winY: y0, winW: sw, winH: sh,
    offsetAtBound: !o.freeOffset && off >= offUb * (1 - 1e-6),
    sigmaOk, brightnessOk,
    ok: fit.status > 0 && sigmaOk && brightnessOk,
  };
}

/**
 * PsyFit's integrate_sif on an already-cropped pps image: detect, refine, fit, filter.
 * @returns {{fits: object[], candidates: number, thresholdAbs: number}}
 */
export function detectAndFit(img, w, h, options = {}) {
  const o = { threshold: 10, minDistance: 5, signal: 'UCNP', ...FIT_DEFAULTS, ...options };
  const { coords, thresholdAbs } = detectPeaks(img, w, h, o);
  const fits = [];
  let limit = o.maxPeaks ?? 5000;
  for (const [cy, cx] of coords) {
    if (limit-- <= 0) break;
    const f = fitPsf(img, w, h, Math.trunc(cx), Math.trunc(cy), o);
    if (f && (f.ok || o.keepRejected)) fits.push(f);
  }
  return { fits, candidates: coords.length, thresholdAbs };
}

/** Drop fits that converged onto the same spot (keeps the first of each cluster). */
export function dedupeFits(fits, minSep = 1.0) {
  const out = [];
  for (const f of fits) {
    let dup = false;
    for (const k of out) {
      if (Math.abs(k.x - f.x) < minSep && Math.abs(k.y - f.y) < minSep) { dup = true; break; }
    }
    if (!dup) out.push(f);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Crowded fields: PsyFit's fit_psfs_crowded. Each PSF is fitted on raw pixels
// together with every detection whose light reaches its window (a sum of
// Gaussians over one shared background), then a second pass looks for peaks
// left in the residual.
// ---------------------------------------------------------------------------

const pyRound = (v) => {            // Python's round(): half to even
  const f = Math.floor(v), d = v - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
};

function percentileLinear(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function crowdedPass(img, w, h, pts, radiusPix, pixUm, sigmaUbUm) {
  const r = Math.trunc(radiusPix), reach = r + 3;
  const sigLb = 0.3, sigUb = Math.max(sigmaUbUm / pixUm, 0.6);
  const sig0 = Math.min(0.15 / pixUm, sigUb * 0.8);
  const rows = [];
  for (let i = 0; i < pts.length; i++) {
    const cy = pts[i][0], cx = pts[i][1];
    const y0 = Math.trunc(Math.max(pyRound(cy) - r, 0)), y1 = Math.trunc(Math.min(pyRound(cy) + r + 1, h));
    const x0 = Math.trunc(Math.max(pyRound(cx) - r, 0)), x1 = Math.trunc(Math.min(pyRound(cx) + r + 1, w));
    const sw = x1 - x0, sh = y1 - y0;
    if (sw <= 0 || sh <= 0 || sw * sh < 9) continue;
    const sub = new Float64Array(sw * sh);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) sub[y * sw + x] = img[(y0 + y) * w + x0 + x];
    const order = [i];
    for (let j = 0; j < pts.length; j++) {
      if (j !== i && Math.abs(pts[j][0] - cy) <= reach && Math.abs(pts[j][1] - cx) <= reach) order.push(j);
    }
    const sorted = Float64Array.from(sub).sort();
    const bg0 = percentileLinear(sorted, 0.10);
    const subMax = sorted[sorted.length - 1];
    const peakMax = subMax > bg0 ? subMax - bg0 : 1.0;
    const K = order.length, n = 5 * K + 1;
    const p0 = new Float64Array(n), lb = new Float64Array(n), ub = new Float64Array(n);
    order.forEach((j, k) => {
      const py = pts[j][0], px = pts[j][1];
      const iy = Math.min(h - 1, Math.max(0, pyRound(py))), ix = Math.min(w - 1, Math.max(0, pyRound(px)));
      const a0 = Math.max(img[iy * w + ix] - bg0, 1e-3);
      p0.set([a0, px, py, sig0, sig0], 5 * k);
      lb.set([0, px - 1, py - 1, sigLb, sigLb], 5 * k);
      ub.set([Math.max(3 * a0, 2 * peakMax, 1.0), px + 1, py + 1, sigUb, sigUb], 5 * k);
    });
    p0[n - 1] = bg0; lb[n - 1] = Math.min(0, bg0);
    ub[n - 1] = Math.max(percentileLinear(sorted, 0.5), bg0 + 1e-6);
    for (let k = 0; k < n; k++) p0[k] = Math.min(ub[k] - 1e-9, Math.max(lb[k] + 1e-9, p0[k]));

    const grad = new Float64Array(n);
    const model = (p, H, g) => {
      let cost = 0;
      for (let y = 0; y < sh; y++) {
        const py = y0 + y;
        for (let x = 0; x < sw; x++) {
          const pxx = x0 + x;
          let f = p[n - 1];
          for (let k = 0; k < K; k++) {
            const A = p[5 * k], ux = pxx - p[5 * k + 1], uy = py - p[5 * k + 2];
            const sx = p[5 * k + 3], sy = p[5 * k + 4];
            const e = Math.exp(-(ux * ux / (2 * sx * sx) + uy * uy / (2 * sy * sy)));
            f += A * e;
            if (H) {
              const ge = A * e;
              grad[5 * k] = e;
              grad[5 * k + 1] = ge * ux / (sx * sx);
              grad[5 * k + 2] = ge * uy / (sy * sy);
              grad[5 * k + 3] = ge * ux * ux / (sx * sx * sx);
              grad[5 * k + 4] = ge * uy * uy / (sy * sy * sy);
            }
          }
          const res = f - sub[y * sw + x];
          cost += res * res;
          if (H) {
            grad[n - 1] = 1;
            for (let a = 0; a < n; a++) {
              const ga = grad[a];
              g[a] += ga * res;
              if (ga === 0) continue;
              const row = a * n;
              for (let b = a; b < n; b++) H[row + b] += ga * grad[b];
            }
          }
        }
      }
      if (H) for (let a = 1; a < n; a++) for (let b = 0; b < a; b++) H[a * n + b] = H[b * n + a];
      return cost;
    };
    const fit = trfLeastSquares(model, p0, lb, ub, { m: sw * sh, xScale: 'jac', maxNfev: 100 * n });
    const p = fit.x;
    // Light inside the window that belongs to this PSF alone.
    let integrated = 0;
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        let others = 0;
        for (let k = 1; k < K; k++) {
          const ux = x0 + x - p[5 * k + 1], uy = y0 + y - p[5 * k + 2];
          others += p[5 * k] * Math.exp(-(ux * ux / (2 * p[5 * k + 3] ** 2) + uy * uy / (2 * p[5 * k + 4] ** 2)));
        }
        integrated += sub[y * sw + x] - p[n - 1] - others;
      }
    }
    rows.push({ A: p[0], x: p[1], y: p[2], sx: p[3], sy: p[4], offset: p[n - 1], neighbors: K - 1, integrated,
      winX: x0, winY: y0, winW: sw, winH: sh });
  }
  return rows;
}

/**
 * @param {Array<[number, number]>} coords detections as [row, col]
 * @returns {object[]} fits shaped like fitPsf's results (plus `neighbors`)
 */
export function fitCrowded(img, w, h, coords, options = {}) {
  const o = { ...FIT_DEFAULTS, minDistance: 3, passes: 2, thresholdAbs: null, ...options };
  const pix = o.pixUm;
  const radiusPix = Math.max(1, Math.trunc(o.radiusUm / pix));
  let pts = coords.map((c) => [Number(c[0]), Number(c[1])]);
  if (!pts.length) return [];
  for (let pass = 1; pass < Math.max(1, o.passes); pass++) {
    if (o.thresholdAbs === null || o.thresholdAbs === undefined) break;
    const first = crowdedPass(img, w, h, pts, radiusPix, pix, o.sigmaUbUm);
    if (!first.length) break;
    const resid = Float32Array.from(img);
    for (const f of first) {
      const rr = Math.trunc(4 * Math.max(f.sx, f.sy)) + 1;
      const ya = Math.max(Math.trunc(f.y) - rr, 0), yb = Math.min(Math.trunc(f.y) + rr + 1, h);
      const xa = Math.max(Math.trunc(f.x) - rr, 0), xb = Math.min(Math.trunc(f.x) + rr + 1, w);
      for (let y = ya; y < yb; y++) {
        for (let x = xa; x < xb; x++) {
          resid[y * w + x] -= f.A * Math.exp(-((x - f.x) ** 2 / (2 * f.sx * f.sx) + (y - f.y) ** 2 / (2 * f.sy * f.sy)));
        }
      }
    }
    const sm = gaussianFilter(resid, w, h, 1);
    let extra = peakLocalMax(sm, w, h, { minDistance: Math.max(Math.trunc(o.minDistance), 1), thresholdAbs: o.thresholdAbs });
    const known = first.map((f) => [f.y, f.x]);
    extra = extra.filter(([ey, ex]) => {
      let dmin = Infinity;
      for (const [ky, kx] of known) { const dd = Math.hypot(ey - ky, ex - kx); if (dd < dmin) dmin = dd; }
      return dmin >= 2.0;
    });
    if (!extra.length) break;
    pts = known.concat(extra);
  }
  const rows = crowdedPass(img, w, h, pts, radiusPix, pix, o.sigmaUbUm);
  let fits = [];
  for (const f of rows) {
    const sigmaOk = f.sx * pix <= o.sigThresholdUm + 1e-3 && f.sy * pix <= o.sigThresholdUm + 1e-3;
    const brightnessOk = f.integrated >= 1 && f.integrated <= 1e9;
    if (!(sigmaOk && brightnessOk) && !o.keepRejected) continue;
    fits.push({
      x: f.x, y: f.y, peakX: f.x, peakY: f.y, amp: f.A, sx: f.sx, sy: f.sy, offset: f.offset,
      brightness: f.integrated, brightnessFit: 2 * Math.PI * f.A * f.sx * f.sy,
      r2: NaN, noise: NaN, snr: NaN, neighbors: f.neighbors,
      window: f.winW * f.winH, winX: f.winX, winY: f.winY, winW: f.winW, winH: f.winH,
      sigmaOk, brightnessOk, ok: sigmaOk && brightnessOk,
    });
  }
  if (fits.length > 1) {
    // Two detections on one particle converge to the same spot: keep the brighter.
    fits.sort((a, b) => b.brightnessFit - a.brightnessFit);
    const keep = new Array(fits.length).fill(true);
    for (let a = 0; a < fits.length; a++) {
      if (!keep[a]) continue;
      for (let b = a + 1; b < fits.length; b++) {
        if (keep[b] && Math.hypot(fits[b].x - fits[a].x, fits[b].y - fits[a].y) < 1.0) keep[b] = false;
      }
    }
    fits = fits.filter((_, k) => keep[k]).sort((a, b) => (a.y - b.y) || (a.x - b.x));
  }
  return fits;
}
