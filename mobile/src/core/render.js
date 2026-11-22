// Colormaps, overlay hues, the intensity-to-colour mapping and scene compositing.
//
// A LUT is a Uint8Array of 256 RGB triples. Single images go through a
// colormap LUT; overlays give each layer a black-to-hue ramp and add the
// layers together, so coincident particles show as the mixed colour.

import { applyAffine } from './geometry.js';

// Matplotlib colormaps sampled at 33 evenly spaced points (linear interpolation
// between them is within 6/255 of the originals).
const CMAP_HEX = {
  magma: '0000040303120a0822130d341d114729115a36106b440f7651127c5d177f6a1c81762181832681902a819c2e7faa337db73779c43c75d0416fdc4869e75263ef5d5ef56b5cf9795dfc8961fd9869fea772feb67cfec488fed395fde2a3fcf0b2fcfdbf',
  inferno: '0000040403120b0724150b37210c4a2f0a5b3d09654a0c6b57106e64156e71196e7d1e6d8a226a972766a32c61b0315bbc3754c73e4cd24644db503be45a31eb6628f1731df68013f98e09fb9d07fcac11fbbc21f9cb35f5db4cf2ea69f3f68afcffa4',
  viridis: '440154470d6048186a482374472d7b4537814240863e49893b528b375b8d33638d2f6b8e2c728e297a8e26828e23898e21918c1f988b1fa08822a78528ae8032b67a3fbc734ec36b5ec96270cf5784d44b98d83eaddc30c2df23d8e219ece51bfde725',
  plasma: '0d08872206903105973f049c4c02a15901a56600a77201a87e03a88a09a59511a1a01a9caa2395b32c8ebc3587c43e7fcc4778d35171da5a6ae06363e66c5ceb7655f0804ef58b47f89540fba139fdac33feb82cfdc527fcd225f8df25f4ed27f0f921',
  cividis: '00224e00285b002e6a0533711a386f273e6e32436d3b496c434e6c4b546c535a6d5a5f6e61656f686a716f70737676767d7c788482798c8878938e789b9476a39a74aba072b4a76fbcae6cc4b468cdbb63d5c25edec958e7d150f0d846f9e03afee838',
  hot: '0b00002000003500004a00005f00007400008900009e0000b30000c80000dd0000f20000ff0800ff1d00ff3200ff4700ff5c00ff7100ff8600ff9b00ffb000ffc500ffda00ffef00ffff07ffff26ffff46ffff65ffff85ffffa4ffffc4ffffe3ffffff',
};

/** Colormaps offered for single images, default first. */
export const COLORMAPS = ['grey', 'magma', 'inferno', 'viridis', 'plasma', 'cividis', 'hot'];

// Overlay hues. The first four are PsyFit's colorblind-safe merge palette
// (blue, green, red and pink for NIR); all are bright on black.
const HUES = {
  green: [51, 255, 133],
  magenta: [255, 92, 225],
  cyan: [64, 224, 255],
  yellow: [255, 221, 51],
  red: [255, 42, 0],
  blue: [89, 122, 255],
  orange: [255, 150, 40],
  pink: [255, 128, 172],
  gray: [255, 255, 255],
};
export const HUE_NAMES = Object.keys(HUES);

const hueRgb = (hue) => HUES[hue] || HUES.gray;
export const hueCss = (hue) => { const c = hueRgb(hue); return `rgb(${c[0]}, ${c[1]}, ${c[2]})`; };

const luts = new Map();

function rampLut(rgb) {
  const lut = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    lut[i * 3] = Math.round((rgb[0] * i) / 255);
    lut[i * 3 + 1] = Math.round((rgb[1] * i) / 255);
    lut[i * 3 + 2] = Math.round((rgb[2] * i) / 255);
  }
  return lut;
}

function tableLut(hex) {
  const n = hex.length / 6;
  const pts = new Uint8Array(n * 3);
  for (let i = 0; i < n * 3; i++) pts[i] = parseInt(hex.substr(i * 2, 2), 16);
  const lut = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * (n - 1);
    const a = Math.min(n - 2, Math.floor(pos)), f = pos - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(pts[a * 3 + c] * (1 - f) + pts[(a + 1) * 3 + c] * f);
  }
  return lut;
}

/** LUT for a colormap name ("grey", "magma", ...) or an overlay hue ("green", ...). */
export function lutRgb(name) {
  const key = name === 'gray' ? 'grey' : String(name);
  if (!luts.has(key)) {
    luts.set(key, CMAP_HEX[key] ? tableLut(CMAP_HEX[key]) : rampLut(key === 'grey' ? HUES.gray : hueRgb(key)));
  }
  return luts.get(key);
}

/** CSS gradient of a colormap or hue ramp, dark end first. */
export function lutCss(name, direction = '90deg') {
  const lut = lutRgb(name);
  const stops = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round((i / 8) * 255) * 3;
    stops.push(`rgb(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

/**
 * The display mapping between an intensity and its position 0..1 on the colour
 * scale. Linear between `lo` and `hi`, or logarithmic; a logarithmic scale
 * whose low end is not positive is shifted so that it starts at 1, the way
 * PsyFit plots log images as image + 1.
 * @returns {{fwd: (v: number) => number, inv: (t: number) => number, log: boolean}}
 */
export function makeNorm(lo, hi, log = false) {
  if (!(hi > lo)) hi = lo + 1;
  const clamp = (t) => (t > 0 ? (t < 1 ? t : 1) : 0);     // NaN -> 0
  if (!log) {
    const k = 1 / (hi - lo);
    return { log: false, fwd: (v) => clamp((v - lo) * k), inv: (t) => lo + t * (hi - lo) };
  }
  const off = lo > 0 ? 0 : 1 - lo;
  const a = lo + off, b = hi + off;
  const k = 1 / Math.log(b / a);
  return {
    log: true,
    fwd: (v) => (v + off > 0 ? clamp(Math.log((v + off) / a) * k) : 0),
    inv: (t) => a * Math.pow(b / a, t) - off,
  };
}

let accum = null;

/**
 * Draw a scene (see store.buildScene) into RGBA pixels of scene.w x scene.h.
 * Output row 0 is the TOP of the picture; image row 0 is drawn at the bottom,
 * as PsyFit shows it.
 * @param {object} scene
 * @param {Uint8ClampedArray} out length scene.w * scene.h * 4
 */
export function composeScene(scene, out) {
  const W = scene.w, H = scene.h;
  const layers = scene.layers;

  // One untransformed layer: straight through its colormap.
  if (layers.length === 1 && !layers[0].inv && layers[0].rect.w === W && layers[0].rect.h === H) {
    const L = layers[0];
    const { data, sw, rect, lut } = L;
    const fwd = makeNorm(L.lo, L.hi, L.log).fwd;
    const lin = !L.log, lo = L.lo, k = 255 / ((L.hi > L.lo ? L.hi : L.lo + 1) - L.lo);
    for (let Y = 0; Y < H; Y++) {
      let s = (rect.y + Y) * sw + rect.x;
      let d = (H - 1 - Y) * W * 4;
      for (let X = 0; X < W; X++, s++, d += 4) {
        let i;
        if (lin) {
          const t = (data[s] - lo) * k;
          i = t > 0 ? (t < 255 ? (t + 0.5) | 0 : 255) : 0;
        } else {
          i = (fwd(data[s]) * 255 + 0.5) | 0;
        }
        i *= 3;
        out[d] = lut[i]; out[d + 1] = lut[i + 1]; out[d + 2] = lut[i + 2]; out[d + 3] = 255;
      }
    }
    return;
  }

  // Several layers, or a shifted one: resample each onto the scene and add.
  const n = W * H * 3;
  if (!accum || accum.length !== n) accum = new Float32Array(n);
  else accum.fill(0);
  for (const L of layers) {
    const { data, sw, rect, lut, inv } = L;
    const fwd = makeNorm(L.lo, L.hi, L.log).fwd;
    const op = L.opacity === undefined ? 1 : L.opacity;
    const rw = rect.w, rh = rect.h;
    for (let Y = 0; Y < H; Y++) {
      let d = (H - 1 - Y) * W * 3;
      for (let X = 0; X < W; X++, d += 3) {
        let v;
        if (inv) {
          const p = applyAffine(inv, X, Y);
          const lx = p[0], ly = p[1];
          if (!(lx >= 0 && ly >= 0 && lx <= rw - 1 && ly <= rh - 1)) continue;
          const x0 = Math.min(rw - 2, lx | 0), y0 = Math.min(rh - 2, ly | 0);
          const fx = lx - x0, fy = ly - y0;
          const r0 = (rect.y + y0) * sw + rect.x + x0, r1 = r0 + sw;
          v = (data[r0] * (1 - fx) + data[r0 + 1] * fx) * (1 - fy) + (data[r1] * (1 - fx) + data[r1 + 1] * fx) * fy;
        } else {
          if (X >= rw || Y >= rh) continue;
          v = data[(rect.y + Y) * sw + rect.x + X];
        }
        const i = ((fwd(v) * 255 + 0.5) | 0) * 3;
        accum[d] += lut[i] * op; accum[d + 1] += lut[i + 1] * op; accum[d + 2] += lut[i + 2] * op;
      }
    }
  }
  for (let p = 0, d = 0; p < n; p += 3, d += 4) {
    out[d] = accum[p]; out[d + 1] = accum[p + 1]; out[d + 2] = accum[p + 2]; out[d + 3] = 255;
  }
}
