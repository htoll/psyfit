// Particle tracking: link per-frame localizations into trajectories and
// summarise their motion (mean-squared displacement, diffusion coefficient).

/**
 * Link detections into tracks.
 *
 * Frame by frame, every (open track, detection) pair closer than the allowed
 * displacement is a candidate; candidates are taken shortest-first so each
 * track and detection is used once (a global greedy assignment, which is near
 * optimal when particles are sparser than their step length). A track may skip
 * up to `maxGap` frames (blinking / missed detection) before it is closed; the
 * search radius grows with the square root of the gap, as for diffusion.
 *
 * @param {Array<Array<{x:number, y:number}>>} frames detections per frame
 * @param {{maxDisp?:number, maxGap?:number, minLength?:number}} opts pixels / frames
 * @returns {Array<{id:number, points:Array<object>}>} each point is the
 *   detection object with an added `f` (frame index)
 */
export function linkTracks(frames, opts = {}) {
  const maxDisp = opts.maxDisp ?? 5;
  const maxGap = Math.max(0, opts.maxGap ?? 2);
  const minLength = Math.max(1, opts.minLength ?? 3);
  const open = [];
  const done = [];
  let nextId = 1;
  for (let f = 0; f < frames.length; f++) {
    const dets = frames[f] || [];
    const cands = [];
    for (let t = 0; t < open.length; t++) {
      const tr = open[t];
      const last = tr.points[tr.points.length - 1];
      const gap = f - last.f;
      const reach = maxDisp * Math.sqrt(gap);
      for (let d = 0; d < dets.length; d++) {
        const dx = dets[d].x - last.x, dy = dets[d].y - last.y;
        if (Math.abs(dx) > reach || Math.abs(dy) > reach) continue;
        const dist = Math.hypot(dx, dy);
        if (dist <= reach) cands.push([dist / Math.sqrt(gap), t, d]);
      }
    }
    cands.sort((a, b) => a[0] - b[0]);
    const usedT = new Uint8Array(open.length), usedD = new Uint8Array(dets.length);
    for (const [, t, d] of cands) {
      if (usedT[t] || usedD[d]) continue;
      usedT[t] = 1; usedD[d] = 1;
      open[t].points.push({ ...dets[d], f });
    }
    for (let d = 0; d < dets.length; d++) {
      if (!usedD[d]) open.push({ id: nextId++, points: [{ ...dets[d], f }] });
    }
    for (let t = open.length - 1; t >= 0; t--) {
      const last = open[t].points[open[t].points.length - 1];
      if (f - last.f > maxGap) done.push(open.splice(t, 1)[0]);
    }
  }
  const all = done.concat(open).filter((t) => t.points.length >= minLength);
  all.sort((a, b) => (a.points[0].f - b.points[0].f) || (a.id - b.id));
  all.forEach((t, i) => { t.id = i + 1; });
  return all;
}

/**
 * Time-averaged mean-squared displacement.
 * @returns {Array<{lag:number, msd:number, n:number}>} lag in frames, msd in px^2
 */
export function trackMsd(points, maxLag) {
  const n = points.length;
  if (n < 2) return [];
  const span = points[n - 1].f - points[0].f;
  const top = Math.max(1, Math.min(maxLag ?? Math.floor(span / 2), span));
  const sums = new Float64Array(top + 1), counts = new Uint32Array(top + 1);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const lag = points[j].f - points[i].f;
      if (lag > top) break;
      const dx = points[j].x - points[i].x, dy = points[j].y - points[i].y;
      sums[lag] += dx * dx + dy * dy;
      counts[lag]++;
    }
  }
  const out = [];
  for (let lag = 1; lag <= top; lag++) if (counts[lag]) out.push({ lag, msd: sums[lag] / counts[lag], n: counts[lag] });
  return out;
}

/**
 * Diffusion coefficient from the first lags of the MSD:  MSD = 4 D t + 4 s^2.
 * @param {number} dt seconds per frame
 * @param {number} pixUm micrometres per pixel
 * @returns {null | {D:number, locPrecisionUm:number, alpha:number, lagsUsed:number}} D in um^2/s
 */
export function fitDiffusion(msd, dt, pixUm, nFit = 4) {
  const use = msd.slice(0, Math.max(2, nFit)).filter((m) => m.n > 0);
  if (use.length < 2) return null;
  const t = use.map((m) => m.lag * dt), y = use.map((m) => m.msd * pixUm * pixUm);
  const n = t.length;
  let mt = 0, my = 0;
  for (let i = 0; i < n; i++) { mt += t[i]; my += y[i]; }
  mt /= n; my /= n;
  let stt = 0, sty = 0;
  for (let i = 0; i < n; i++) { stt += (t[i] - mt) ** 2; sty += (t[i] - mt) * (y[i] - my); }
  if (!(stt > 0)) return null;
  const slope = sty / stt, intercept = my - slope * mt;
  // Anomalous exponent from a log-log fit over all available lags.
  let alpha = NaN;
  const lg = msd.filter((m) => m.msd > 0 && m.n > 0).slice(0, Math.max(4, Math.floor(msd.length / 2)));
  if (lg.length >= 3) {
    const lx = lg.map((m) => Math.log(m.lag * dt)), ly = lg.map((m) => Math.log(m.msd));
    let ax = 0, ay = 0;
    for (let i = 0; i < lx.length; i++) { ax += lx[i]; ay += ly[i]; }
    ax /= lx.length; ay /= lx.length;
    let sxx = 0, sxy = 0;
    for (let i = 0; i < lx.length; i++) { sxx += (lx[i] - ax) ** 2; sxy += (lx[i] - ax) * (ly[i] - ay); }
    if (sxx > 0) alpha = sxy / sxx;
  }
  return { D: slope / 4, locPrecisionUm: Math.sqrt(Math.max(intercept, 0) / 4), alpha, lagsUsed: n };
}

/** Per-track summary in physical units. */
export function trackStats(track, dt, pixUm) {
  const p = track.points;
  let path = 0;
  for (let i = 1; i < p.length; i++) path += Math.hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
  const net = Math.hypot(p[p.length - 1].x - p[0].x, p[p.length - 1].y - p[0].y);
  const frames = p[p.length - 1].f - p[0].f;
  const msd = trackMsd(p, Math.max(1, Math.min(25, Math.floor(frames / 2))));
  const diff = fitDiffusion(msd, dt, pixUm);
  let bsum = 0, bn = 0;
  for (const q of p) if (Number.isFinite(q.brightness)) { bsum += q.brightness; bn++; }
  return {
    id: track.id, points: p.length, firstFrame: p[0].f, lastFrame: p[p.length - 1].f,
    durationS: frames * dt,
    pathUm: path * pixUm, netUm: net * pixUm,
    speedUmS: frames > 0 ? (path * pixUm) / (frames * dt) : 0,
    straightness: path > 0 ? net / path : 0,
    D: diff ? diff.D : NaN, alpha: diff ? diff.alpha : NaN,
    locPrecisionUm: diff ? diff.locPrecisionUm : NaN,
    meanBrightness: bn ? bsum / bn : NaN,
    msd,
  };
}

/** A stable, well-spread colour for a track id (golden-angle hue walk). */
export function trackColor(id) {
  const hue = (id * 137.508) % 360;
  const light = 58 + ((id * 7) % 3) * 6;
  return `hsl(${hue.toFixed(1)}, 92%, ${light}%)`;
}
