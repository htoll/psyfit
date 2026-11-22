// Descriptive statistics, histogram binning, 1-D Gaussian mixtures, linear
// regression and Kalafut-Visscher step detection (ported from PsyFit's utils.py).

export function finiteValues(values) {
  const out = [];
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) out.push(v); }
  return out;
}

export function mean(a) {
  if (!a.length) return NaN;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

export function std(a, ddof = 0) {
  if (a.length - ddof <= 0) return NaN;
  const m = mean(a);
  let v = 0;
  for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2;
  return Math.sqrt(v / (a.length - ddof));
}

/** Linear-interpolated quantile of an ascending-sorted array (numpy default). */
export function quantileSorted(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function median(a) {
  return quantileSorted(Float64Array.from(a).sort(), 0.5);
}

/**
 * Min/max, mean and approximate percentiles of an image, from one pass plus a
 * 2048-bin histogram (fast enough to run on every frame change).
 */
export function imageStats(data, step = 1) {
  let min = Infinity, max = -Infinity, sum = 0, n = 0;
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (v !== v) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v; n++;
  }
  if (!n) return { min: 0, max: 1, mean: 0, n: 0, percentile: () => 0, bins: new Uint32Array(1) };
  const NB = 2048;
  const bins = new Uint32Array(NB);
  const span = max - min || 1;
  const k = (NB - 1) / span;
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (v !== v) continue;
    bins[((v - min) * k) | 0]++;
  }
  const percentile = (p) => {
    const target = (p / 100) * n;
    let acc = 0;
    for (let b = 0; b < NB; b++) {
      acc += bins[b];
      if (acc >= target) return min + ((b + 0.5) / NB) * span;
    }
    return max;
  };
  return { min, max, mean: sum / n, n, percentile, bins };
}

// ---------------------------------------------------------------------------
// Histogram binning (numpy "auto")
// ---------------------------------------------------------------------------

export function autoBinCount(values) {
  const x = Float64Array.from(values).sort();
  const n = x.length;
  if (n < 2) return 1;
  const range = x[n - 1] - x[0];
  if (!(range > 0)) return 1;
  const iqr = quantileSorted(x, 0.75) - quantileSorted(x, 0.25);
  const fd = 2 * iqr * Math.pow(n, -1 / 3);
  const sturges = range / (Math.log2(n) + 1);
  const sqrt = range / Math.sqrt(n);
  const width = fd ? Math.min(Math.max(fd, sqrt / 2), sturges) : sturges;
  if (!(width > 0)) return 1;
  return Math.max(1, Math.min(400, Math.ceil(range / width)));
}

/** @returns {{edges: Float64Array, counts: Uint32Array, width: number}} */
export function histogram(values, bins, lo, hi) {
  const n = Math.max(1, bins | 0);
  if (lo === undefined || hi === undefined) {
    lo = Infinity; hi = -Infinity;
    for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
  const edges = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) edges[i] = lo + (hi - lo) * i / n;
  const counts = new Uint32Array(n);
  const k = n / (hi - lo);
  for (const v of values) {
    if (!(v >= lo && v <= hi)) continue;
    let b = Math.floor((v - lo) * k);
    if (b >= n) b = n - 1;
    counts[b]++;
  }
  return { edges, counts, width: (hi - lo) / n };
}

// ---------------------------------------------------------------------------
// 1-D Gaussian mixture (EM). Deterministic: the same data always gives the
// same fit. Several starting points are tried and the most likely fit kept.
// ---------------------------------------------------------------------------

const SQRT2PI = Math.sqrt(2 * Math.PI);

function emRun(x, k, means, sigmas, weights, maxIter = 300, tol = 1e-7) {
  const n = x.length;
  const resp = new Float64Array(n * k);
  let prev = -Infinity, ll = -Infinity;
  const floor = 1e-6;
  for (let iter = 0; iter < maxIter; iter++) {
    ll = 0;
    for (let i = 0; i < n; i++) {
      let tot = 0;
      for (let c = 0; c < k; c++) {
        const z = (x[i] - means[c]) / sigmas[c];
        const p = weights[c] * Math.exp(-0.5 * z * z) / (sigmas[c] * SQRT2PI);
        resp[i * k + c] = p;
        tot += p;
      }
      if (!(tot > 0)) {
        // Far from every component: assign to the nearest.
        let best = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { const d = Math.abs(x[i] - means[c]) / sigmas[c]; if (d < bd) { bd = d; best = c; } }
        for (let c = 0; c < k; c++) resp[i * k + c] = c === best ? 1 : 0;
        ll += -745;
      } else {
        for (let c = 0; c < k; c++) resp[i * k + c] /= tot;
        ll += Math.log(tot);
      }
    }
    for (let c = 0; c < k; c++) {
      let nk = 0, mu = 0;
      for (let i = 0; i < n; i++) { const r = resp[i * k + c]; nk += r; mu += r * x[i]; }
      if (nk < 1e-10) { weights[c] = 1e-10; continue; }
      mu /= nk;
      let va = 0;
      for (let i = 0; i < n; i++) va += resp[i * k + c] * (x[i] - mu) ** 2;
      means[c] = mu;
      sigmas[c] = Math.sqrt(va / nk + floor);
      weights[c] = nk / n;
    }
    if (Math.abs(ll - prev) <= tol * Math.abs(ll)) break;
    prev = ll;
  }
  return { means, sigmas, weights, logLik: ll };
}

/**
 * @param {ArrayLike<number>} values
 * @param {number} k components
 * @returns {null | {k:number, n:number, means:number[], sigmas:number[], weights:number[], logLik:number, bic:number}}
 *   components sorted by mean; null when there are too few points.
 */
export function gmm1d(values, k = 2) {
  const x = Float64Array.from(finiteValues(values)).sort();
  const n = x.length;
  k = Math.max(1, Math.floor(k));
  if (n <= k) return null;
  const sAll = std(x) || Math.abs(x[0]) * 1e-3 || 1e-3;
  // Work in standardised units so the variance floor is scale-free.
  const m0 = mean(x);
  const z = x.map((v) => (v - m0) / sAll);
  const starts = [];
  // (a) equal-count quantile blocks
  {
    const mu = [], sg = [], w = [];
    for (let c = 0; c < k; c++) {
      const a = Math.floor((c * n) / k), b = Math.max(a + 1, Math.floor(((c + 1) * n) / k));
      const seg = z.subarray(a, b);
      mu.push(mean(seg)); sg.push(Math.max(std(seg), 0.05)); w.push(1 / k);
    }
    starts.push([mu, sg, w]);
  }
  // (b) split at the k-1 largest gaps between neighbours
  if (k > 1) {
    const gaps = [];
    for (let i = 1; i < n; i++) gaps.push([z[i] - z[i - 1], i]);
    gaps.sort((a, b) => b[0] - a[0]);
    const cuts = gaps.slice(0, k - 1).map((g) => g[1]).sort((a, b) => a - b);
    const bounds = [0, ...cuts, n];
    const mu = [], sg = [], w = [];
    for (let c = 0; c < k; c++) {
      const seg = z.subarray(bounds[c], bounds[c + 1]);
      mu.push(mean(seg)); sg.push(Math.max(std(seg) || 0, 0.05)); w.push(seg.length / n);
    }
    starts.push([mu, sg, w]);
  }
  // (c) evenly spaced means across the range
  if (k > 1) {
    const mu = [], sg = [], w = [];
    for (let c = 0; c < k; c++) {
      mu.push(z[0] + (z[n - 1] - z[0]) * (c + 0.5) / k);
      sg.push(Math.max((z[n - 1] - z[0]) / (2 * k), 0.05)); w.push(1 / k);
    }
    starts.push([mu, sg, w]);
  }
  let best = null;
  for (const [mu, sg, w] of starts) {
    const r = emRun(z, k, Float64Array.from(mu), Float64Array.from(sg), Float64Array.from(w));
    if (!best || r.logLik > best.logLik) best = r;
  }
  const order = Array.from({ length: k }, (_, i) => i).sort((a, b) => best.means[a] - best.means[b]);
  const logLik = best.logLik - n * Math.log(sAll);
  return {
    k, n,
    means: order.map((c) => best.means[c] * sAll + m0),
    sigmas: order.map((c) => best.sigmas[c] * sAll),
    weights: order.map((c) => best.weights[c]),
    logLik,
    bic: -2 * logLik + (3 * k - 1) * Math.log(n),
  };
}

export function gmmPdf(g, v) {
  let p = 0;
  for (let c = 0; c < g.k; c++) {
    const zz = (v - g.means[c]) / g.sigmas[c];
    p += g.weights[c] * Math.exp(-0.5 * zz * zz) / (g.sigmas[c] * SQRT2PI);
  }
  return p;
}

/**
 * Which component summarises the population (PsyFit's `pick`):
 * "weight" = the most populated; "lowest" = the dimmest holding at least
 * `minWeight` of the particles (the single-particle peak).
 */
export function gmmPick(g, pick = 'weight', minWeight = 0.15) {
  let idx = 0;
  for (let c = 1; c < g.k; c++) if (g.weights[c] > g.weights[idx]) idx = c;
  if (pick === 'lowest') {
    for (let c = 0; c < g.k; c++) if (g.weights[c] >= minWeight) { idx = c; break; }
  }
  return { index: idx, mu: g.means[idx], sigma: g.sigmas[idx], weight: g.weights[idx] };
}

// ---------------------------------------------------------------------------
// Linear regression (scipy.stats.linregress)
// ---------------------------------------------------------------------------

export function linregress(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 2) return null;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  if (!(sxx > 0)) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r = syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
  return { slope, intercept, r, r2: r * r, n };
}

// ---------------------------------------------------------------------------
// Kalafut-Visscher step detection (Comput. Phys. Commun. 179 (2008) 716).
// Piecewise-constant model; a new step edge is kept only if it lowers the BIC.
// ---------------------------------------------------------------------------

function bestTwoLevelSplit(y, start, end) {
  const n = end - start;
  if (n < 2) return [null, Infinity];
  let total = 0, total2 = 0;
  for (let i = start; i < end; i++) { total += y[i]; total2 += y[i] * y[i]; }
  let best = Infinity, bestI = null, ls = 0, l2 = 0;
  for (let i = 1; i < n; i++) {
    const v = y[start + i - 1];
    ls += v; l2 += v * v;
    const rs = total - ls, r2 = total2 - l2;
    const sse = (l2 - ls * ls / i) + (r2 - rs * rs / (n - i));
    if (sse < best) { best = sse; bestI = start + i; }
  }
  return [bestI, best];
}

function piecewiseSse(y, bounds) {
  let sse = 0;
  for (let s = 0; s < bounds.length - 1; s++) {
    const a = bounds[s], b = bounds[s + 1];
    if (b <= a) continue;
    let m = 0;
    for (let i = a; i < b; i++) m += y[i];
    m /= (b - a);
    for (let i = a; i < b; i++) sse += (y[i] - m) ** 2;
  }
  return sse;
}

/**
 * @param {ArrayLike<number>} values per-frame signal (non-finite samples are dropped)
 * @param {number} penalty multiplier on the BIC complexity term (>1 = fewer steps)
 * @returns {number[]} segment boundaries including both ends, e.g. [0, k1, k2, N]
 */
export function detectStepsKV(values, penalty = 1.0) {
  const y = Float64Array.from(finiteValues(values));
  const N = y.length;
  if (N < 4) return [0, N];
  const bic = (bounds) => {
    const sse = Math.max(piecewiseSse(y, bounds), 1e-12);
    return N * Math.log(sse / N) + penalty * bounds.length * Math.log(N);
  };
  let bounds = [0, N];
  let cur = bic(bounds);
  for (;;) {
    let best = null;
    for (let s = 0; s < bounds.length - 1; s++) {
      const [idx] = bestTwoLevelSplit(y, bounds[s], bounds[s + 1]);
      if (idx === null || bounds.includes(idx)) continue;
      const cand = bounds.concat([idx]).sort((a, b) => a - b);
      const cb = bic(cand);
      if (!best || cb < best[0]) best = [cb, cand];
    }
    if (best && best[0] < cur - 1e-9) { cur = best[0]; bounds = best[1]; } else break;
  }
  return bounds;
}

/** Plateau level for every sample plus one record per step edge. */
export function staircaseFromBounds(values, bounds) {
  const y = Float64Array.from(finiteValues(values));
  const levels = new Float64Array(y.length).fill(NaN);
  const means = [];
  for (let s = 0; s < bounds.length - 1; s++) {
    const a = bounds[s], b = bounds[s + 1];
    let m = 0;
    for (let i = a; i < b; i++) m += y[i];
    m = b > a ? m / (b - a) : NaN;
    for (let i = a; i < b; i++) levels[i] = m;
    means.push(m);
  }
  const steps = [];
  for (let j = 1; j < means.length; j++) {
    steps.push({ index: bounds[j], before: means[j - 1], after: means[j], drop: means[j - 1] - means[j] });
  }
  return { levels, steps, means };
}

// ---------------------------------------------------------------------------
// Small deterministic PRNG (demo data, RANSAC, colours)
// ---------------------------------------------------------------------------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianSampler(rand) {
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0, r = 0;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r);
    spare = v * f;
    return u * f;
  };
}
