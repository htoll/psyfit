// Simulated acquisitions, written as real .sif bytes so the demo goes through
// exactly the same loading path as files from the microscope. Clearly labelled
// as simulated wherever they appear in the UI.

import { writeSif } from './sif.js';
import { mulberry32, gaussianSampler } from './stats.js';
import { buildCalibration, SPEC_LEFT } from './spectra.js';

function addPsf(img, w, h, x, y, amp, sx, sy = sx) {
  const r = Math.ceil(4 * Math.max(sx, sy));
  const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(w - 1, Math.ceil(x + r));
  const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(h - 1, Math.ceil(y + r));
  for (let yy = y0; yy <= y1; yy++) {
    const ey = Math.exp(-0.5 * ((yy - y) / sy) ** 2);
    for (let xx = x0; xx <= x1; xx++) {
      img[yy * w + xx] += amp * ey * Math.exp(-0.5 * ((xx - x) / sx) ** 2);
    }
  }
}

function addNoise(img, gauss, readNoise, excess) {
  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    img[i] = Math.max(0, Math.round(v + Math.sqrt(Math.max(v, 0) * excess) * gauss() + readNoise * gauss()));
  }
}

/** Channel-to-channel misalignment used by the simulated MCL image (local px). */
export const DEMO_CHANNEL_AFFINES = {
  blue: [1.0030, 0.0020, 2.6, -0.0020, 1.0030, -3.1],
  green: [1, 0, 0, 0, 1, 0],
  red: [0.9980, -0.0030, -1.8, 0.0030, 0.9980, 2.4],
  nir: [1.0010, 0.0010, 3.4, -0.0010, 1.0010, 1.2],
};

/**
 * A 512x512 four-channel (MCL-style) frame: the same particles appear in every
 * quadrant with channel-dependent brightness, PSF width and a small affine
 * misalignment; each quadrant shows its circular aperture.
 */
export function makeDemoImage(seed = 11) {
  const W = 512, H = 512, Qs = 256;
  const rand = mulberry32(seed), gauss = gaussianSampler(rand);
  const img = new Float32Array(W * H).fill(4400);
  const quad = { blue: [0, 256], green: [256, 256], red: [0, 0], nir: [256, 0] };
  const rel = { blue: 0.55, green: 1.0, red: 0.72, nir: 0.3 };
  const sig = { blue: 1.14, green: 1.24, red: 1.38, nir: 1.62 };
  const centre = { blue: [129.5, 126.0], green: [127.0, 129.0], red: [130.5, 128.5], nir: [126.5, 127.0] };
  const R = 117;
  for (const ch of Object.keys(quad)) {
    const [qx, qy] = quad[ch];
    const [cx, cy] = centre[ch];
    for (let y = 0; y < Qs; y++) {
      for (let x = 0; x < Qs; x++) {
        const d = Math.hypot(x - cx, y - cy);
        // soft-edged illuminated disc
        const edge = 1 / (1 + Math.exp((d - R) / 1.2));
        img[(qy + y) * W + qx + x] += 260 * edge * (1 - 0.25 * (d / R) ** 2);
      }
    }
  }
  const particles = [];
  let guard = 0;
  while (particles.length < 46 && guard++ < 5000) {
    const a = rand() * Math.PI * 2, rr = Math.sqrt(rand()) * (R - 16);
    const x = 127.5 + rr * Math.cos(a), y = 128 + rr * Math.sin(a);
    if (particles.some((p) => Math.hypot(p.x - x, p.y - y) < 9)) continue;
    const u = rand();
    const order = u < 0.66 ? 1 : u < 0.9 ? 2 : u < 0.97 ? 3 : 5;
    particles.push({ x, y, order, jitter: Math.exp(0.11 * gauss()) });
  }
  const monomerAmp = 3000;   // counts at the peak in the green channel
  for (const p of particles) {
    for (const ch of Object.keys(quad)) {
      const m = DEMO_CHANNEL_AFFINES[ch];
      // green is the reference; other channels see the particle displaced by the inverse map
      const det = m[0] * m[4] - m[1] * m[3];
      const ux = p.x - m[2], uy = p.y - m[5];
      const lx = (m[4] * ux - m[1] * uy) / det, ly = (-m[3] * ux + m[0] * uy) / det;
      const amp = monomerAmp * p.order * p.jitter * rel[ch] * Math.exp(0.05 * gauss()) * (1.24 / sig[ch]) ** 2;
      addPsf(img, W, H, quad[ch][0] + lx, quad[ch][1] + ly, amp, sig[ch]);
    }
  }
  addNoise(img, gauss, 22, 1.6);
  return {
    name: 'Demo_simulated_MCL_976nm_4channel.sif',
    bytes: writeSif({
      width: W, height: H, frames: [img], exposure: 0.05, cycle: 0.051742, gain: 50, accumulations: 10,
      temperature: -75, originalName: 'C:\\Demo\\simulated_MCL\\Demo_simulated_MCL_976nm_4channel.sif',
    }),
    truth: { particles, affines: DEMO_CHANNEL_AFFINES },
  };
}

/**
 * A 256x256, 80-frame movie: freely diffusing particles plus a few immobile
 * ones that bleach in steps or blink.
 */
export function makeDemoMovie(seed = 5, nFrames = 80) {
  const W = 256, H = 256;
  const rand = mulberry32(seed), gauss = gaussianSampler(rand);
  const dt = 0.05, pix = 0.1011;
  const movers = [];
  for (let i = 0; i < 14; i++) {
    const D = 0.03 + rand() * 0.32;                         // um^2/s
    movers.push({ x: 24 + rand() * (W - 48), y: 24 + rand() * (H - 48), D, step: Math.sqrt(2 * D * dt) / pix, amp: 520 + rand() * 700 });
  }
  const fixed = [
    { x: 60.3, y: 58.6, levels: [[0, 3], [22, 2], [47, 1], [66, 0]], unit: 420 },   // three bleaching steps
    { x: 196.7, y: 70.2, levels: [[0, 2], [35, 1], [58, 0]], unit: 460 },
    { x: 78.4, y: 201.1, levels: [[0, 1], [18, 0], [29, 1], [44, 0], [52, 1]], unit: 640 },  // blinking
    { x: 190.2, y: 188.4, levels: [[0, 1]], unit: 900 },
  ];
  const frames = [];
  const truth = movers.map((m) => ({ D: m.D, path: [] }));
  for (let f = 0; f < nFrames; f++) {
    const img = new Float32Array(W * H).fill(310);
    movers.forEach((m, i) => {
      if (f > 0) {
        m.x += m.step * gauss(); m.y += m.step * gauss();
        if (m.x < 12) m.x = 24 - m.x; if (m.x > W - 12) m.x = 2 * (W - 12) - m.x;
        if (m.y < 12) m.y = 24 - m.y; if (m.y > H - 12) m.y = 2 * (H - 12) - m.y;
      }
      truth[i].path.push([m.x, m.y]);
      addPsf(img, W, H, m.x, m.y, m.amp, 1.3);
    });
    for (const s of fixed) {
      let level = 0;
      for (const [start, lv] of s.levels) if (f >= start) level = lv;
      if (level > 0) addPsf(img, W, H, s.x, s.y, s.unit * level, 1.28);
    }
    addNoise(img, gauss, 9, 1.8);
    frames.push(img);
  }
  return {
    name: 'Demo_simulated_tracking_movie_80frames.sif',
    bytes: writeSif({
      width: W, height: H, frames, exposure: 0.045, cycle: dt, gain: 300, accumulations: 1,
      temperature: -75, originalName: 'C:\\Demo\\simulated\\Demo_simulated_tracking_movie_80frames.sif',
    }),
    truth: { movers: truth, fixed, dt, pix },
  };
}

// ---------------------------------------------------------------------------
// Nikon: a localization image at the top left and its spectra at the bottom right
// ---------------------------------------------------------------------------

/** Emission lines of the simulated particles (nm) and how the two simulated kinds weight them. */
export const DEMO_NIKON_LINES = [475, 545, 660, 800];
const DEMO_NIKON_KINDS = [[0.25, 1.0, 0.55, 0.2], [1.0, 0.3, 0.15, 0.75]];
/** wavelength(i) = a * exp(b * i) + c for sample i of a spectrum's window (values typical of the real instrument). */
const DEMO_NIKON_FIT = [56.6, 0.0221, 376.4];

/** Where the spectrum of a particle at flipped-frame position (x, y) is anchored: nearly a shift, slightly distorted. */
function demoNikonMap(x, y) {
  const u = (x - 140) / 140, v = (y - 140) / 140;
  return [1.037 * x - 0.003 * y + 190.4 + 2.2 * u * v, -0.023 * x + 1.001 * y + 250.5 + 1.6 * (u * u - v * v)];
}

/**
 * A 512x512 Nikon-style frame with the calibration that describes it.
 * Coordinates in `truth` are in the flipped frame PsyFit's spectral tools use
 * (x = column, y = 511 - row). Two particles are placed so that their spectra
 * run into each other.
 */
export function makeDemoNikon(seed = 23) {
  const W = 512, H = 512;
  const rand = mulberry32(seed), gauss = gaussianSampler(rand);
  const img = new Float32Array(W * H).fill(4800);
  const put = (x, yf, amp, sx, sy) => addPsf(img, W, H, x, H - 1 - yf, amp, sx, sy);
  const [fa, fb, fc] = DEMO_NIKON_FIT;
  const sampleOf = (nm) => Math.log((nm - fc) / fa) / fb;
  // soft-edged illuminated discs: the field, and its dispersed copy
  const cx = 140, cy = 140, R = 122;
  const [sxc, syc] = demoNikonMap(cx, cy);
  for (let yf = 0; yf < H; yf++) {
    for (let x = 0; x < W; x++) {
      const d = Math.hypot(x - cx, yf - cy);
      const ds = Math.hypot((x - (sxc - 42)) / 1.18, yf - syc);
      img[(H - 1 - yf) * W + x] += 420 / (1 + Math.exp((d - R) / 1.5)) + 300 / (1 + Math.exp((ds - R) / 2.5));
    }
  }
  const particles = [];
  const add = (x, y) => {
    const kind = rand() < 0.5 ? 0 : 1;
    particles.push({ x, y, kind, amp: (9000 + 26000 * rand()) * Math.exp(0.15 * gauss()) });
  };
  add(118.3, 96.6); add(131.7, 98.1);               // 13 px apart along x, same rows: their spectra overlap
  let guard = 0;
  while (particles.length < 34 && guard++ < 4000) {
    const a = rand() * Math.PI * 2, rr = Math.sqrt(rand()) * (R - 18);
    const x = cx + rr * Math.cos(a), y = cy + rr * Math.sin(a);
    if (particles.some((p) => Math.hypot(p.x - x, p.y - y) < 10)) continue;
    add(x, y);
  }
  for (const p of particles) {
    put(p.x, p.y, p.amp, 1.25, 1.25);
    const [xs, ys] = demoNikonMap(p.x, p.y);
    p.xs = xs; p.ys = ys;
    DEMO_NIKON_LINES.forEach((nm, k) => {
      put(xs - SPEC_LEFT + sampleOf(nm), ys, 0.16 * p.amp * DEMO_NIKON_KINDS[p.kind][k], 2.1, 1.3);
    });
  }
  addNoise(img, gauss, 30, 1.6);
  // the calibration: a grid of points over the field, each with its spectral position; every other one has a wavelength fit
  const grid = new Map(), fits = new Map();
  let n = 0;
  for (let gy = 22; gy <= 262; gy += 13) {
    for (let gx = 22 + ((gy / 13) % 2 ? 6 : 0); gx <= 262; gx += 13) {
      if (Math.hypot(gx - cx, gy - cy) > R + 6) continue;
      const id = `P${String(n).padStart(4, '0')}`;
      const jx = gx + 0.6 * (rand() - 0.5), jy = gy + 0.6 * (rand() - 0.5);
      grid.set(id, ['sim', [jx, jy], Float64Array.from([30000, jx, jy, 3, 3, 0, 24000]), demoNikonMap(jx, jy), new Float64Array(100)]);
      if (n % 2 === 0) fits.set(id, DEMO_NIKON_FIT.slice());
      n++;
    }
  }
  return {
    name: 'Demo_simulated_Nikon_976nm_spectra.sif',
    bytes: writeSif({
      width: W, height: H, frames: [img], exposure: 0.1, cycle: 0.101742, gain: 300, accumulations: 10,
      temperature: -75, originalName: 'C:\\Demo\\simulated_Nikon\\Demo_simulated_Nikon_976nm_spectra.sif',
    }),
    calibration: buildCalibration(grid, fits, { name: 'Simulated demo calibration', fitsName: 'Simulated demo fits', date: '', demo: true }),
    truth: { particles, lines: DEMO_NIKON_LINES, kinds: DEMO_NIKON_KINDS, sampleOf },
  };
}
