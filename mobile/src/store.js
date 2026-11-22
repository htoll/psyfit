// Application state and every action that changes it.
//
// Conventions: image coordinates are (x = column, y = row) with row 0 first, as
// in PsyFit's x_pix / y_pix. "Sensor" coordinates index the full frame; a
// channel's "local" coordinates start at its rectangle; "scene" coordinates are
// what the viewer shows (the base layer's local frame).

import { openSif } from './core/sif.js';
import { detectAndFit, detectPeaks, fitCrowded, fitPsf, gaussianFilter } from './core/numeric.js';
import { imageStats, gmm1d, gmmPick, linregress, detectStepsKV, staircaseFromBounds, median } from './core/stats.js';
import {
  IDENTITY, applyAffine, composeAffine, invertAffine, isIdentity, translation,
  estimateRegistration, matchPoints, fitApertures, describeAffine,
} from './core/geometry.js';
import { linkTracks, trackStats } from './core/tracking.js';
import { SCOPES, guessScope, pixelSize, channelRect, availableChannels, channelAt, channelInfo, cropRect } from './core/layouts.js';
import {
  readCalibrationFiles, calibrationFromJson, calibrationToJson, mapToSpectrum, wavelengthAxis, sampleSpectrum, objectPoint,
  spectraOverlap, offDetector, spectralWindow, SPEC_NPTS,
} from './core/spectra.js';
import { lutRgb, HUE_NAMES } from './core/render.js';
import { makeDemoImage, makeDemoMovie, makeDemoNikon } from './core/demo.js';
import { loadPrefs, savePrefs, fileCache, CACHE_MAX_FILE, calStore, drive, driveErrorText, CAL_MAX_BYTES } from './platform.js';

const prefs = loadPrefs();
const pick = (obj, defaults) => {
  const out = { ...defaults };
  if (obj && typeof obj === 'object') {
    for (const k of Object.keys(defaults)) if (typeof obj[k] === typeof defaults[k]) out[k] = obj[k];
  }
  return out;
};

export const S = {
  files: [],
  activeId: null,
  layers: [],
  overlay: false,
  scopeDefault: SCOPES[prefs.scopeDefault] ? prefs.scopeDefault : 'mcl',
  display: pick(prefs.display, {
    cmap: 'grey', log: false, showFits: true, showLabels: true, showScaleBar: true, showColorbar: true, showGrid: true,
    useReg: true, perFrameAuto: false, colorBy: 'none',
  }),
  fit: pick(prefs.fit, { threshold: 10, minDistance: 5, sigMaxUm: 0.3, signal: 'UCNP', crowded: false, live: true }),
  hist: pick(prefs.hist, { components: 2, pick: 'weight', pool: 'file', bins: 0 }),
  movie: pick(prefs.movie, { fps: 12, loop: true, tail: 20, colorMode: 'random', showTracks: true, budgetMB: 160 }),
  track: pick(prefs.track, { maxDisp: 5, maxGap: 2, minLength: 5 }),
  coloc: { a: null, b: null, radius: 2, result: null },
  registration: prefs.registration && typeof prefs.registration === 'object' ? prefs.registration : {},
  sel: null,            // { ids: number[] } | { trackId, fileId }
  sheet: null,          // 'files' | 'display' | 'analyze' | 'overlay' | 'movie'
  plot: null,           // { kind: 'hist' | 'scatter' | 'trace' | 'msd' | 'table', ... }
  trace: null,
  busy: null,           // { label, done, total, cancel }
  toast: null,          // { text, kind }
  playing: false,
  inspect: null,        // live pixel read-out
  scale: false,         // colour-scale editor open (tap the colour bar)
  // Nikon spectra: the calibration in use, Drive search results, and whether possible overlaps are shown
  spec: { cal: null, busy: '', found: null, showOverlaps: false },
};
{
  const saved = calStore.all();
  if (saved.length) S.spec.cal = calibrationFromJson(saved[0]);
}

// ---------------------------------------------------------------------------
// Change notification
// ---------------------------------------------------------------------------

const listeners = new Set();
let flags = 0, scheduled = false;
const FLAG = { ui: 1, overlay: 2, scene: 4 };

export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

/** kind: 'ui' (panels only), 'overlay' (also redraw markers), 'scene' (also recompose pixels). */
export function bump(kind = 'ui') {
  flags |= kind === 'scene' ? 7 : kind === 'overlay' ? 3 : 1;
  if (scheduled) return;
  scheduled = true;
  Promise.resolve().then(() => {
    scheduled = false;
    const f = flags; flags = 0;
    listeners.forEach((fn) => { try { fn({ scene: !!(f & FLAG.scene), overlay: !!(f & FLAG.overlay) }); } catch (e) { console.error(e); } });
  });
}

let persistTimer = null;
export function persist() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    savePrefs({
      scopeDefault: S.scopeDefault, display: S.display, fit: S.fit, hist: S.hist, movie: S.movie,
      track: S.track, registration: S.registration,
    });
  }, 300);
}

let toastTimer = null;
export function toast(text, kind = 'info', ms = 3200) {
  S.toast = { text, kind };
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { S.toast = null; bump(); }, kind === 'error' ? Math.max(ms, 6000) : ms);
  bump();
}

const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------
// Files and frames
// ---------------------------------------------------------------------------

let seq = 0;
export const fileById = (id) => S.files.find((f) => f.id === id) || null;
export const activeFile = () => fileById(S.activeId);

function toPps(raw, info) {
  // Same float32 steps as PsyFit: counts * (5 / gain) / exposure / accumulations.
  const gain = info.GainDAC > 0 ? info.GainDAC : 1;
  const k = 5.0 / gain;
  const e = info.ExposureTime > 0 ? info.ExposureTime : 1;
  const a = info.AccumulatedCycles > 0 ? info.AccumulatedCycles : 1;
  for (let i = 0; i < raw.length; i++) raw[i] = Math.fround(Math.fround(raw[i] * k) / e) / a;
  return raw;
}

export function maxFramesFor(file) {
  return Math.max(1, Math.floor((S.movie.budgetMB * 1e6) / (file.src.width * file.src.height * 4)));
}

function makeEntry(src, origin) {
  const scope = guessScope(src.name, src.info) || S.scopeDefault;
  return {
    id: `f${++seq}`, name: src.name, src, origin: origin || { kind: 'file' },
    scope, scopeGuessed: !!guessScope(src.name, src.info),
    pixUm: pixelSize(scope, src.name, src.info),
    isMovie: src.frames > 1,
    frames: new Map(), loaded: [], cur: 0, view: 'frame', window: null,
    proj: {}, stats: new Map(),
    fits: [], fitsKey: null, manual: [], tracks: null,
  };
}

/**
 * Load a set of frames (replacing those in memory).
 * @param {{start:number, step:number, count:number}} win 0-based start frame
 */
export async function loadWindow(file, win) {
  const total = file.src.frames;
  const step = Math.max(1, Math.floor(win.step || 1));
  const start = Math.min(total - 1, Math.max(0, Math.floor(win.start || 0)));
  const room = Math.floor((total - 1 - start) / step) + 1;
  const count = Math.max(1, Math.min(Math.floor(win.count || 1), room, maxFramesFor(file)));
  const wanted = [];
  for (let i = 0; i < count; i++) wanted.push(start + i * step);
  stopPlayback();
  file.frames = new Map(); file.loaded = []; file.cur = 0; file.proj = {}; file.stats = new Map();
  file.tracks = null; file.fits = []; file.fitsKey = null; file.view = 'frame';
  if (S.trace && S.trace.fileId === file.id) S.trace = null;
  const job = { cancelled: false };
  const showProgress = count > 3;
  if (showProgress) { S.busy = { label: `Loading ${file.name}`, done: 0, total: count, cancel: () => { job.cancelled = true; } }; bump(); }
  let last = performance.now();
  try {
    for (let i = 0; i < wanted.length; i++) {
      if (job.cancelled) break;
      const raw = await file.src.readFrame(wanted[i]);
      file.frames.set(wanted[i], toPps(raw, file.src.info));
      file.loaded.push(wanted[i]);
      if (showProgress && performance.now() - last > 60) {
        S.busy.done = i + 1; last = performance.now(); bump();
        await tick();
      }
    }
  } finally {
    if (showProgress) S.busy = null;
  }
  file.window = { start, step, count: file.loaded.length };
  if (!file.loaded.length) throw new Error('No frames could be read');
  bump('scene');
}

/**
 * Open .sif blobs. Each item: { blob, name, origin }.
 * @returns {Promise<number>} how many opened
 */
export async function addFiles(items) {
  let opened = 0, lastId = null;
  for (const item of items) {
    try {
      const src = await openSif(item.blob, item.name);
      const entry = makeEntry(src, item.origin);
      if (entry.scope === 'nikon') chooseCalibrationFor(entry);
      const n = entry.isMovie ? Math.min(src.frames, maxFramesFor(entry)) : 1;
      await loadWindow(entry, { start: 0, step: 1, count: n });
      S.files.push(entry);
      lastId = entry.id; opened++;
      if (src.truncated) toast(`${entry.name}: the file ends early, ${src.frames} of ${src.info.frames} frames are present.`, 'warn');
    } catch (err) {
      toast(`${item.name}: ${err && err.message ? err.message : 'could not be opened'}`, 'error');
    }
  }
  if (lastId) {
    const firstNew = S.files[S.files.length - opened];
    setActive(opened > 1 ? firstNew.id : lastId);
  }
  bump('scene');
  return opened;
}

/** Open files chosen with the system picker (and remember small ones on this device). */
export async function addPickedFiles(fileList) {
  const items = [];
  for (const f of Array.from(fileList || [])) {
    items.push({ blob: f, name: f.name, origin: { kind: 'file' } });
    if (f.size <= CACHE_MAX_FILE) {
      f.arrayBuffer().then((buf) => fileCache.put(`local:${f.name}:${f.size}`, f.name, new Uint8Array(buf), 'Files')).catch(() => {});
    }
  }
  return addFiles(items);
}

export async function addBytes(name, bytes, origin, cacheKey) {
  if (cacheKey) fileCache.put(cacheKey, name, bytes, origin && origin.kind === 'drive' ? 'Google Drive' : 'Files').catch(() => {});
  return addFiles([{ blob: new Blob([bytes]), name, origin }]);
}

export async function openCached(key) {
  const row = await fileCache.get(key);
  if (!row) { toast('That file is no longer stored on this device.', 'warn'); return 0; }
  return addFiles([{ blob: new Blob([row.buf]), name: row.name, origin: { kind: 'cache' } }]);
}

export async function loadDemo(kind) {
  const demo = kind === 'movie' ? makeDemoMovie() : kind === 'nikon' ? makeDemoNikon() : makeDemoImage();
  const before = S.scopeDefault;
  if (kind === 'movie') S.scopeDefault = 'full';
  const n = await addFiles([{ blob: new Blob([demo.bytes]), name: demo.name, origin: { kind: 'demo' } }]);
  S.scopeDefault = before;
  // The simulated Nikon frame comes with the calibration it was simulated with.
  if (n && demo.calibration) setCalibration(demo.calibration);
  return n;
}

export function removeFile(id) {
  const i = S.files.findIndex((f) => f.id === id);
  if (i < 0) return;
  stopPlayback();
  S.files.splice(i, 1);
  S.layers = S.layers.filter((l) => l.fileId !== id);
  if (S.trace && S.trace.fileId === id) S.trace = null;
  S.coloc.result = null; S.sel = null;
  if (S.activeId === id) {
    S.activeId = null;
    if (S.files.length) setActive(S.files[Math.min(i, S.files.length - 1)].id);
  }
  if (!S.layers.length && S.files.length) setActive(S.files[0].id);
  bump('scene');
}

export function frameKey(file) {
  if (file.view !== 'frame') return file.view;
  return `f${file.loaded[file.cur]}`;
}

/** The pps image currently shown for a file (a frame or a projection). */
export function frameData(file) {
  if (!file || !file.loaded.length) return null;
  if (file.view === 'mean' || file.view === 'max') return projection(file, file.view);
  return file.frames.get(file.loaded[Math.min(file.cur, file.loaded.length - 1)]) || null;
}

function projection(file, kind) {
  if (file.proj[kind]) return file.proj[kind];
  const n = file.src.width * file.src.height;
  const out = new Float32Array(n);
  if (kind === 'max') out.fill(-Infinity);
  for (const idx of file.loaded) {
    const fr = file.frames.get(idx);
    if (kind === 'max') { for (let i = 0; i < n; i++) if (fr[i] > out[i]) out[i] = fr[i]; }
    else { for (let i = 0; i < n; i++) out[i] += fr[i]; }
  }
  if (kind === 'mean') { const k = 1 / file.loaded.length; for (let i = 0; i < n; i++) out[i] *= k; }
  file.proj[kind] = out;
  return out;
}

export function setView(file, view) {
  file.view = view;
  stopPlayback();
  afterImageChange(file);
}

export function setFrame(file, pos, fromPlayback = false) {
  const p = Math.max(0, Math.min(file.loaded.length - 1, pos));
  if (p === file.cur && file.view === 'frame') return;
  file.cur = p; file.view = 'frame';
  if (!fromPlayback) afterImageChange(file); else bump('scene');
}

function afterImageChange(file) {
  S.sel = S.sel && S.sel.trackId ? S.sel : null;
  bump('scene');
  if (S.fit.live && !S.playing && file.id === S.activeId) scheduleDetection();
}

export function setScope(file, scopeId) {
  if (!SCOPES[scopeId] || file.scope === scopeId) return;
  file.scope = scopeId; file.scopeGuessed = true;
  file.pixUm = pixelSize(scopeId, file.name, file.src.info);
  if (scopeId === 'nikon') chooseCalibrationFor(file);
  file.fits = []; file.manual = []; file.tracks = null; file.stats = new Map();
  S.scopeDefault = scopeId;
  S.layers = S.layers.filter((l) => l.fileId !== file.id);
  S.coloc.result = null; S.sel = null;
  if (file.id === S.activeId || !S.layers.length) { S.overlay = S.overlay && S.layers.length > 0; setActive(file.id); }
  persist();
  bump('scene');
}

export function setPixelSize(file, um) {
  if (!(um > 0)) return;
  file.pixUm = um;
  bump('overlay');
  if (S.fit.live) scheduleDetection();
}

// ---------------------------------------------------------------------------
// Layers, channels and the scene
// ---------------------------------------------------------------------------

let layerSeq = 0;
function newLayer(file, channel) {
  let hue = channelInfo(file.scope, channel).hue || 'gray';
  const used = new Set(S.layers.map((l) => l.hue));
  if (used.has(hue)) hue = ['green', 'magenta', 'cyan', 'yellow', 'red', 'blue', 'orange', 'gray'].find((h) => !used.has(h)) || hue;
  return { id: `l${++layerSeq}`, fileId: file.id, channel, hue, lo: 0, hi: 1, auto: true, visible: true, nudge: [0, 0] };
}

export function setActive(id) {
  const file = fileById(id);
  if (!file) return;
  stopPlayback();
  const prev = S.layers.find((l) => l.fileId === S.activeId);
  S.activeId = id;
  if (!S.overlay || !S.layers.length) {
    let ch = prev ? prev.channel : 'all';
    if (ch !== 'all' && !channelRect(file.scope, ch, file.src)) ch = 'all';
    S.overlay = false;
    S.layers = [];
    S.layers = [newLayer(file, ch)];
  }
  S.sel = null;
  bump('scene');
  if (S.fit.live) scheduleDetection();
}

/** Channel chip tapped for the active file. */
export function selectChannel(channelId, fileId = S.activeId) {
  const file = fileById(fileId);
  if (!file) return;
  if (!S.overlay) {
    S.activeId = file.id;
    S.layers = [];
    S.layers = [newLayer(file, channelId)];
  } else if (channelId === 'all') {
    S.overlay = false; S.activeId = file.id;
    S.layers = [];
    S.layers = [newLayer(file, 'all')];
  } else {
    const i = S.layers.findIndex((l) => l.fileId === file.id && l.channel === channelId);
    if (i >= 0) {
      if (S.layers.length > 1) S.layers.splice(i, 1);
    } else {
      S.layers = S.layers.filter((l) => l.channel !== 'all');
      S.layers.push(newLayer(file, channelId));
    }
  }
  S.sel = null; S.coloc.result = null;
  bump('scene');
  if (S.fit.live) scheduleDetection();
}

/** Add or remove a (file, channel) layer from the overlay. */
export function toggleLayer(fileId, channel) {
  const file = fileById(fileId);
  if (!file) return;
  const i = S.layers.findIndex((l) => l.fileId === fileId && l.channel === channel);
  if (i >= 0) {
    if (S.layers.length > 1) S.layers.splice(i, 1);
  } else {
    if (channel !== 'all') {
      // a whole-frame view of a multi-channel file cannot be mixed with single channels
      S.layers = S.layers.filter((l) => {
        const f = fileById(l.fileId);
        return !(l.channel === 'all' && f && availableChannels(f.scope, f.src).length);
      });
    }
    S.layers.push(newLayer(file, channel));
    S.overlay = true;
  }
  S.sel = null; S.coloc.result = null;
  bump('scene');
  if (S.fit.live) scheduleDetection();
}

export function setOverlay(on) {
  const file = activeFile();
  if (!file || on === S.overlay) return;
  if (on) {
    const chans = availableChannels(file.scope, file.src);
    if (!chans.length) { toast('This file has no channel split. Choose MCL or Nikon under Display, or add another file as a layer.', 'warn'); S.overlay = true; bump(); return; }
    S.overlay = true;
    const cur = S.layers[0];
    if (!cur || cur.channel === 'all') {
      S.layers = [];
      for (const c of chans) S.layers.push(newLayer(file, c.id));
    }
  } else {
    S.overlay = false;
    const keep = S.layers.find((l) => l.fileId === file.id) || S.layers[0];
    S.layers = keep ? [keep] : [];
    if (!S.layers.length) S.layers = [newLayer(file, 'all')];
  }
  S.sel = null; S.coloc.result = null;
  bump('scene');
  if (S.fit.live) scheduleDetection();
}

export function updateLayer(id, patch) {
  const l = S.layers.find((x) => x.id === id);
  if (!l) return;
  Object.assign(l, patch);
  bump('scene');
}

export function removeLayer(id) {
  if (S.layers.length <= 1) return;
  S.layers = S.layers.filter((l) => l.id !== id);
  S.coloc.result = null;
  bump('scene');
}

function regMatrix(scopeId, channel) {
  if (!S.display.useReg) return IDENTITY;
  const reg = S.registration[scopeId];
  const entry = reg && reg.channels && reg.channels[channel];
  return entry && Array.isArray(entry.m) && entry.m.length === 6 ? entry.m : IDENTITY;
}

/** Affine taking a layer's channel-local coordinates to scene coordinates. */
export function layerMatrix(layer, base) {
  const f = fileById(layer.fileId), bf = fileById(base.fileId);
  let m = IDENTITY;
  if (f && bf && layer.channel !== 'all' && base.channel !== 'all' && !(layer.channel === base.channel && f.scope === bf.scope)) {
    const invB = invertAffine(regMatrix(bf.scope, base.channel)) || IDENTITY;
    m = composeAffine(invB, regMatrix(f.scope, layer.channel));
  }
  if (layer.nudge[0] || layer.nudge[1]) m = composeAffine(translation(layer.nudge[0], layer.nudge[1]), m);
  return m;
}

function statsFor(file, channel, rect) {
  const fixed = file.isMovie && !S.display.perFrameAuto && file.view === 'frame';
  const key = `${fixed ? 'ref' : frameKey(file)}:${channel}`;
  if (file.stats.has(key)) return file.stats.get(key);
  const data = fixed ? file.frames.get(file.loaded[0]) : frameData(file);
  const st = imageStats(channel === 'all' ? data : cropRect(data, file.src.width, rect));
  if (file.stats.size > 64) file.stats.clear();
  file.stats.set(key, st);
  return st;
}

export function layerStats(layer) {
  const file = fileById(layer.fileId);
  if (!file) return null;
  const rect = channelRect(file.scope, layer.channel, file.src);
  return rect ? statsFor(file, layer.channel, rect) : null;
}

export function autoRange(st, multi) {
  const lo = multi ? st.percentile(50) : st.percentile(1);
  const hi = Math.max(st.percentile(99.9), lo + 0.2 * (st.max - lo));
  return { lo, hi: hi > lo ? hi : lo + 1 };
}

/** Everything the viewer needs to draw: scene size and the layers with their pixels. */
export function buildScene() {
  const vis = S.layers.filter((l) => l.visible && fileById(l.fileId));
  if (!vis.length) return null;
  const base = vis[0];
  const bf = fileById(base.fileId);
  const baseRect = channelRect(bf.scope, base.channel, bf.src) || { x: 0, y: 0, w: bf.src.width, h: bf.src.height };
  const multi = vis.length > 1;
  const scene = { w: baseRect.w, h: baseRect.h, sensor: !multi && base.channel === 'all', multi, base, baseFile: bf, layers: [] };
  for (const l of vis) {
    const f = fileById(l.fileId);
    const data = frameData(f);
    const rect = channelRect(f.scope, l.channel, f.src);
    if (!data || !rect) continue;
    const m = l === base && !l.nudge[0] && !l.nudge[1] ? IDENTITY : layerMatrix(l, base);
    if (l.auto) { const r = autoRange(statsFor(f, l.channel, rect), multi); l.lo = r.lo; l.hi = r.hi; }
    scene.layers.push({
      layer: l, file: f, data, sw: f.src.width, rect, m,
      inv: isIdentity(m) ? null : invertAffine(m),
      lo: l.lo, hi: l.hi, log: S.display.log,
      lut: multi ? lutRgb(l.hue) : lutRgb(S.display.cmap), opacity: 1,
    });
  }
  return scene.layers.length ? scene : null;
}

/** Colour used for a channel's markers and chart marks. */
export function channelColor(file, channel) {
  return channelInfo(file ? file.scope : 'full', channel).color;
}

// ---------------------------------------------------------------------------
// Detection and fitting
// ---------------------------------------------------------------------------

let fitSeq = 0;
const sigmaUb = (channel) => (channel === 'nir' ? 0.8 : 0.5);   // PsyFit widens the bound for region 4 (NIR)

function toSensorFit(f, rect, file, channel, extra) {
  return {
    ...f,
    x: f.x + rect.x, y: f.y + rect.y,
    lx: f.x, ly: f.y, lpx: f.peakX, lpy: f.peakY,
    peakX: f.peakX + rect.x, peakY: f.peakY + rect.y,
    channel, fileId: file.id, id: ++fitSeq, ...extra,
  };
}

function fitOptions(file, channel) {
  const p = S.fit;
  return {
    threshold: p.threshold, minDistance: p.minDistance, signal: p.signal,
    pixUm: file.pixUm, sigThresholdUm: p.sigMaxUm, sigmaUbUm: sigmaUb(channel),
  };
}

function analyseChannel(file, data, channel) {
  const rect = channelRect(file.scope, channel, file.src);
  if (!rect) return { fits: [], candidates: 0 };
  const img = channel === 'all' ? data : cropRect(data, file.src.width, rect);
  const o = fitOptions(file, channel);
  let fits, candidates;
  if (S.fit.crowded) {
    const det = detectPeaks(img, rect.w, rect.h, o);
    candidates = det.coords.length;
    fits = fitCrowded(img, rect.w, rect.h, det.coords.slice(0, 1500), { ...o, thresholdAbs: det.thresholdAbs });
  } else {
    const r = detectAndFit(img, rect.w, rect.h, { ...o, maxPeaks: 3000 });
    fits = r.fits; candidates = r.candidates;
  }
  return {
    candidates,
    fits: fits.map((q) => {
      const rec = toSensorFit(q, rect, file, channel, { run: channel });
      if (channel === 'all') rec.channel = channelAt(file.scope, file.src, rec.peakX, rec.peakY) || 'border';
      return rec;
    }),
  };
}

/** Channels to analyse for a file: its visible layers, else whatever the active file shows. */
function analysisChannels(file) {
  let chans = S.layers.filter((l) => l.visible && l.fileId === file.id).map((l) => l.channel);
  if (!chans.length) {
    const act = S.layers.filter((l) => l.visible && l.fileId === S.activeId).map((l) => l.channel);
    chans = act.filter((c) => channelRect(file.scope, c, file.src));
  }
  if (file.scope === 'nikon') {
    // Particles are only ever looked for in the localization image (as PsyFit
    // does); the other region holds their spectra, which are streaks, not PSFs.
    return channelRect('nikon', 'loc', file.src) ? ['loc'] : [];
  }
  return chans.length ? Array.from(new Set(chans)) : ['all'];
}

/** Detect and fit every PSF in the shown image(s). */
export function runDetection(opts = {}) {
  const targets = opts.allFiles ? S.files.slice() : [activeFile()].filter(Boolean);
  let total = 0, candidates = 0;
  for (const file of targets) {
    const data = frameData(file);
    if (!data) continue;
    const key = frameKey(file);
    if (file.fitsKey !== key) { file.fits = []; file.fitsKey = key; }
    for (const ch of analysisChannels(file)) {
      const r = analyseChannel(file, data, ch);
      file.fits = file.fits.filter((q) => q.run !== ch).concat(r.fits);
      total += r.fits.length; candidates += r.candidates;
    }
  }
  S.coloc.result = null;
  if (S.sel && S.sel.ids) S.sel = null;
  bump('overlay');
  return { total, candidates };
}

let detectTimer = null;
export function scheduleDetection(delay = 140) {
  clearTimeout(detectTimer);
  detectTimer = setTimeout(() => {
    if (!S.fit.live || S.playing || S.busy) return;
    try { runDetection(); } catch (e) { console.error(e); }
  }, delay);
}

export function setFitParam(patch) {
  Object.assign(S.fit, patch);
  persist();
  bump();
  if (S.fit.live) scheduleDetection();
}

export function clearFits(file = activeFile()) {
  if (!file) return;
  file.fits = []; file.manual = []; file.fitsKey = null;
  S.sel = null; S.coloc.result = null;
  bump('overlay');
}

/** Automatic fits of a file that belong in a given layer channel (current image only). */
export function fitsForChannel(file, channel) {
  if (file.fitsKey !== frameKey(file)) return [];
  if (channel === 'all') {
    const own = file.fits.filter((q) => q.run === 'all');
    return own.length ? own : file.fits;
  }
  const own = file.fits.filter((q) => q.run === channel);
  return own.length ? own : file.fits.filter((q) => q.run === 'all' && q.channel === channel);
}

function manualForChannel(file, channel) {
  return channel === 'all' ? file.manual : file.manual.filter((q) => q.channel === channel || q.run === channel);
}

/** Fits to draw: [{fit, X, Y, layerIndex}] in scene coordinates. */
export function sceneFits(scene) {
  const out = [];
  scene.layers.forEach((L, li) => {
    const list = fitsForChannel(L.file, L.layer.channel).concat(manualForChannel(L.file, L.layer.channel));
    for (const fit of list) {
      const lx = fit.x - L.rect.x, ly = fit.y - L.rect.y;
      if (lx < -0.5 || ly < -0.5 || lx > L.rect.w - 0.5 || ly > L.rect.h - 0.5) continue;
      const [X, Y] = L.inv ? applyAffine(L.m, lx, ly) : [lx, ly];
      out.push({ fit, X, Y, layerIndex: li });
    }
  });
  return out;
}

export function findFit(id) {
  for (const f of S.files) {
    const hit = f.fits.find((q) => q.id === id) || f.manual.find((q) => q.id === id);
    if (hit) return hit;
  }
  return null;
}

/**
 * Double-tap: fit a PSF at a scene position in every visible layer.
 * `snap` is how far (image px) to look for the local maximum around the tap.
 */
export function probeAt(X, Y, snap = 4) {
  const scene = buildScene();
  if (!scene) return [];
  const group = ++fitSeq;
  const made = [];
  let offLoc = false;
  for (const L of scene.layers) {
    const [lx, ly] = L.inv ? applyAffine(L.inv, X, Y) : [X, Y];
    if (lx < 0 || ly < 0 || lx > L.rect.w - 1 || ly > L.rect.h - 1) continue;
    const file = L.file;
    let ch = L.layer.channel, rect = L.rect, px = lx, py = ly;
    if (file.scope === 'nikon') {
      // Fit in the localization image only, in its own frame, whichever view was tapped.
      const loc = channelRect('nikon', 'loc', file.src);
      const sx = lx + L.rect.x, sy = ly + L.rect.y;
      if (!loc || sx < loc.x || sy < loc.y || sx > loc.x + loc.w - 1 || sy > loc.y + loc.h - 1) { offLoc = true; continue; }
      ch = 'loc'; rect = loc; px = sx - loc.x; py = sy - loc.y;
    }
    const img = ch === 'all' ? L.data : cropRect(L.data, L.sw, rect);
    const [cx, cy] = snapToPeak(img, rect.w, rect.h, px, py, snap);
    const o = fitOptions(file, ch);
    const res = fitPsf(img, rect.w, rect.h, cx, cy, o);
    if (!res) continue;
    const rec = toSensorFit(res, rect, file, ch, { run: ch, manual: true, group, frame: frameKey(file) });
    if (ch === 'all') rec.channel = channelAt(file.scope, file.src, rec.peakX, rec.peakY) || 'border';
    // Re-probing the same spot replaces the old probe.
    file.manual = file.manual.filter((q) => !(q.run === rec.run && Math.hypot(q.x - rec.x, q.y - rec.y) < 1.5));
    file.manual.push(rec);
    made.push(rec);
  }
  if (made.length) S.sel = { ids: made.map((r) => r.id) };
  else toast(offLoc ? 'Double-tap a particle in the localization image (top left). Its spectrum is read from the other region.' : 'No peak to fit there.', 'warn', offLoc ? 3600 : 1800);
  bump('overlay');
  return made;
}

function snapToPeak(img, w, h, x, y, radius) {
  const r = Math.max(1, Math.ceil(radius));
  const x0 = Math.max(1, Math.round(x) - r), x1 = Math.min(w - 2, Math.round(x) + r);
  const y0 = Math.max(1, Math.round(y) - r), y1 = Math.min(h - 2, Math.round(y) + r);
  let best = -Infinity, bx = Math.round(x), by = Math.round(y);
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      if ((xx - x) ** 2 + (yy - y) ** 2 > (r + 0.5) ** 2) continue;
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += img[(yy + dy) * w + xx + dx];
      if (s > best) { best = s; bx = xx; by = yy; }
    }
  }
  return [bx, by];
}

export function removeFit(id) {
  for (const f of S.files) {
    f.manual = f.manual.filter((q) => q.id !== id);
    f.fits = f.fits.filter((q) => q.id !== id);
  }
  if (S.sel && S.sel.ids) {
    const ids = S.sel.ids.filter((i) => i !== id);
    S.sel = ids.length ? { ids } : null;
  }
  bump('overlay');
}

/** Photon-limited localization precision (Thompson et al. 2002, with EM excess noise), in nm. */
export function locPrecisionNm(fit, file) {
  const info = file.src.info;
  const T = (info.ExposureTime > 0 ? info.ExposureTime : 1) * (info.AccumulatedCycles > 0 ? info.AccumulatedCycles : 1);
  const N = fit.brightness * T;
  if (!(N > 0)) return NaN;
  const a = file.pixUm * 1000;
  const s = Math.sqrt(fit.sx * fit.sy) * a;
  const b = (Number.isFinite(fit.noise) ? fit.noise : 0) * T;
  const excess = info.GainDAC > 1 ? 2 : 1;
  const v = excess * (s * s + a * a / 12) / N + (8 * Math.PI * s ** 4 * b * b) / (a * a * N * N);
  return Math.sqrt(v);
}

// ---------------------------------------------------------------------------
// Brightness statistics
// ---------------------------------------------------------------------------

/** Brightness values per channel for the histogram: [{key, label, color, values, files}] */
export function brightnessSets(pool = S.hist.pool) {
  const act = activeFile();
  if (!act) return [];
  const files = pool === 'all' ? S.files : [act];
  const sets = new Map();
  for (const file of files) {
    if (file.fitsKey !== frameKey(file)) continue;
    for (const q of file.fits) {
      if (!q.ok) continue;
      const key = q.channel;
      if (!sets.has(key)) {
        const info = channelInfo(file.scope, key);
        sets.set(key, { key, label: key === 'border' ? 'Border' : info.label, color: info.color, values: [], files: new Set() });
      }
      const s = sets.get(key);
      s.values.push(q.brightness); s.files.add(file.id);
    }
  }
  const order = ['all', ...Object.values(SCOPES).flatMap((s) => s.channels.map((c) => c.id)), 'border'];
  return Array.from(sets.values()).sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
}

export function summarize(values, components = S.hist.components, pickMode = S.hist.pick) {
  const vals = values.filter((v) => Number.isFinite(v));
  if (!vals.length) return null;
  const g = vals.length > components ? gmm1d(vals, components) : null;
  const p = g ? gmmPick(g, pickMode) : null;
  return { n: vals.length, gmm: g, mu: p ? p.mu : median(vals), sigma: p ? p.sigma : NaN, median: median(vals) };
}

/** PsyFit's aggregation classes from the single-particle brightness: <2x, 2-3x, 3-4x, >=4x. */
export function classify(values, single) {
  const counts = [0, 0, 0, 0];
  for (const v of values) {
    if (!(v >= 0)) continue;
    counts[v < 2 * single ? 0 : v < 3 * single ? 1 : v < 4 * single ? 2 : 3]++;
  }
  return counts;
}
export const CLASS_NAMES = ['Monomers', 'Dimers', 'Trimers', 'Multimers'];
export const CLASS_COLORS = ['#199e70', '#3987e5', '#c98500', '#e66767'];

// ---------------------------------------------------------------------------
// Channel registration
// ---------------------------------------------------------------------------

function channelImages(file) {
  const data = frameData(file);
  return availableChannels(file.scope, file.src).map((c) => {
    const rect = channelRect(file.scope, c.id, file.src);
    return { id: c.id, rect, data: cropRect(data, file.src.width, rect), w: rect.w, h: rect.h };
  });
}

function storeRegistration(file, reg) {
  S.registration[file.scope] = reg;
  S.display.useReg = true;
  persist();
  bump('scene');
}

/** Register channels from PSFs that appear in several channels of one image. */
export function registerFromPsfs(file = activeFile(), refWanted = null) {
  if (!file) return null;
  const imgs = channelImages(file);
  if (imgs.length < 2) { toast('This file has a single channel, so there is nothing to register.', 'warn'); return null; }
  const pts = {};
  for (const im of imgs) {
    const o = fitOptions(file, im.id);
    pts[im.id] = detectAndFit(im.data, im.w, im.h, { ...o, maxPeaks: 800 }).fits.map((q) => [q.x, q.y]);
  }
  let ref = refWanted && pts[refWanted] && pts[refWanted].length >= 3 ? refWanted : null;
  if (!ref) ref = imgs.reduce((a, b) => (pts[b.id].length > pts[a.id].length ? b : a)).id;
  const prev = S.registration[file.scope];
  const reg = { ref, method: 'psf', source: file.name, when: Date.now(), channels: {} };
  let okCount = 0;
  for (const im of imgs) {
    if (im.id === ref) { reg.channels[im.id] = { m: IDENTITY.slice(), kind: 'reference', n: pts[ref].length, rms: 0 }; continue; }
    const r = estimateRegistration([{ src: pts[im.id], dst: pts[ref] }], { maxShift: 40, minMatches: 3 });
    if (r) { reg.channels[im.id] = { m: r.m, kind: r.kind, n: r.n, rms: r.rms }; okCount++; }
    else {
      const keep = prev && prev.ref === ref && prev.channels[im.id] ? prev.channels[im.id].m : IDENTITY.slice();
      reg.channels[im.id] = { m: keep, kind: 'failed', n: 0, rms: NaN, found: pts[im.id].length };
    }
  }
  if (!okCount) { toast('No PSFs common to two channels were found. Lower the threshold or use an image where the same particles show in every channel.', 'warn', 5000); return null; }
  storeRegistration(file, reg);
  return reg;
}

/** Rough registration: line up the circular aperture seen in each channel. */
export function registerFromAperture(file = activeFile()) {
  if (!file) return null;
  const imgs = channelImages(file);
  if (imgs.length < 2) { toast('This file has a single channel, so there is nothing to align.', 'warn'); return null; }
  const res = fitApertures(imgs);
  if (res.found < 2) { toast('The aperture edge is not visible in at least two channels of this image.', 'warn', 4500); return null; }
  const refIdx = res.circles.findIndex(Boolean);
  const ref = res.circles[refIdx];
  const reg = { ref: imgs[refIdx].id, method: 'aperture', source: file.name, when: Date.now(), channels: {} };
  imgs.forEach((im, i) => {
    const c = res.circles[i];
    if (i === refIdx) reg.channels[im.id] = { m: IDENTITY.slice(), kind: 'reference', circle: c };
    else if (c) reg.channels[im.id] = { m: translation(ref.cx - c.cx, ref.cy - c.cy), kind: 'aperture', circle: c };
    else reg.channels[im.id] = { m: IDENTITY.slice(), kind: 'failed' };
  });
  storeRegistration(file, reg);
  return reg;
}

export function clearRegistration(scopeId) {
  delete S.registration[scopeId];
  persist();
  bump('scene');
}

export function importRegistration(scopeId, text) {
  const obj = JSON.parse(text);
  if (!obj || typeof obj !== 'object' || !obj.channels) throw new Error('Not a registration file');
  for (const k of Object.keys(obj.channels)) {
    const m = obj.channels[k].m;
    if (!Array.isArray(m) || m.length !== 6 || m.some((v) => !Number.isFinite(v))) throw new Error('Not a registration file');
  }
  S.registration[scopeId] = obj;
  persist();
  bump('scene');
}

export { describeAffine };

/** Shift a layer onto the base layer using PSFs common to both (drift between acquisitions). */
export function alignLayer(layerId) {
  const scene = buildScene();
  if (!scene || scene.layers.length < 2) return null;
  const L = scene.layers.find((x) => x.layer.id === layerId), B = scene.layers[0];
  if (!L || L === B) return null;
  const pts = (X) => {
    const img = X.layer.channel === 'all' ? X.data : cropRect(X.data, X.sw, X.rect);
    const fits = detectAndFit(img, X.rect.w, X.rect.h, { ...fitOptions(X.file, X.layer.channel), maxPeaks: 800 }).fits;
    const saved = X.layer.nudge; X.layer.nudge = [0, 0];
    const m = layerMatrix(X.layer, B.layer);
    X.layer.nudge = saved;
    return fits.map((q) => applyAffine(m, q.x, q.y));
  };
  const r = estimateRegistration([{ src: pts(L), dst: pts(B) }], { maxShift: 40, minMatches: 3, shiftOnly: true });
  if (!r) { toast('Too few PSFs common to both layers to align them.', 'warn'); return null; }
  L.layer.nudge = [r.m[2], r.m[5]];
  S.coloc.result = null;
  bump('scene');
  return r;
}

// ---------------------------------------------------------------------------
// Colocalization between two layers
// ---------------------------------------------------------------------------

export function computeColoc() {
  const scene = buildScene();
  if (!scene || scene.layers.length < 2) { S.coloc.result = null; bump(); return null; }
  const A = scene.layers.find((x) => x.layer.id === S.coloc.a) || scene.layers[0];
  const B = scene.layers.find((x) => x.layer.id === S.coloc.b && x !== A) || scene.layers.find((x) => x !== A);
  const collect = (L) => {
    let list = fitsForChannel(L.file, L.layer.channel);
    if (!list.length) {
      const data = frameData(L.file);
      const key = frameKey(L.file);
      if (L.file.fitsKey !== key) { L.file.fits = []; L.file.fitsKey = key; }
      const r = analyseChannel(L.file, data, L.layer.channel);
      L.file.fits = L.file.fits.filter((q) => q.run !== L.layer.channel).concat(r.fits);
      list = fitsForChannel(L.file, L.layer.channel);
    }
    list = list.filter((q) => q.ok);
    return { list, pts: list.map((q) => applyAffine(L.m, q.x - L.rect.x, q.y - L.rect.y)) };
  };
  const a = collect(A), b = collect(B);
  for (const f of S.files) for (const q of f.fits) q.coloc = false;
  const m = matchPoints(a.pts, b.pts, S.coloc.radius);
  const pairs = m.pairs.map((p) => {
    a.list[p.i].coloc = true; b.list[p.j].coloc = true;
    return { a: a.list[p.i], b: b.list[p.j], d: p.d };
  });
  const xs = pairs.map((p) => p.a.brightness), ys = pairs.map((p) => p.b.brightness);
  S.coloc.a = A.layer.id; S.coloc.b = B.layer.id;
  S.coloc.result = {
    aId: A.layer.id, bId: B.layer.id, nA: a.list.length, nB: b.list.length, pairs,
    fracA: a.list.length ? pairs.length / a.list.length : 0,
    fracB: b.list.length ? pairs.length / b.list.length : 0,
    meanDist: pairs.length ? pairs.reduce((s, p) => s + p.d, 0) / pairs.length : NaN,
    regression: pairs.length >= 2 ? linregress(xs, ys) : null,
  };
  bump('overlay');
  return S.coloc.result;
}

export function layerLabel(layer) {
  const f = fileById(layer.fileId);
  if (!f) return '';
  const ch = channelInfo(f.scope, layer.channel).label;
  const many = new Set(S.layers.map((l) => l.fileId)).size > 1;
  return many ? `${ch} · ${shortName(f.name)}` : ch;
}

export function shortName(name, max = 26) {
  const base = String(name).replace(/\.sif$/i, '');
  return base.length <= max ? base : `${base.slice(0, Math.ceil(max / 2) - 1)}…${base.slice(-Math.floor(max / 2))}`;
}

// ---------------------------------------------------------------------------
// Movies: playback, tracking, intensity traces
// ---------------------------------------------------------------------------

let playTimer = null;
export function stopPlayback() {
  if (!S.playing) return;
  S.playing = false;
  clearTimeout(playTimer);
  bump();
}

export function togglePlayback() {
  const file = activeFile();
  if (!file || file.loaded.length < 2) return;
  if (S.playing) { stopPlayback(); if (S.fit.live) scheduleDetection(); return; }
  S.playing = true;
  file.view = 'frame';
  if (file.cur >= file.loaded.length - 1) file.cur = 0;
  let next = performance.now();
  const step = () => {
    if (!S.playing) return;
    const f = activeFile();
    if (!f || f.loaded.length < 2) { stopPlayback(); return; }
    let p = f.cur + 1;
    if (p >= f.loaded.length) {
      if (!S.movie.loop) { stopPlayback(); return; }
      p = 0;
    }
    setFrame(f, p, true);
    next += 1000 / Math.max(1, S.movie.fps);
    const now = performance.now();
    if (next < now - 250) next = now;
    playTimer = setTimeout(step, Math.max(0, next - now));
  };
  bump();
  playTimer = setTimeout(step, 1000 / Math.max(1, S.movie.fps));
}

/** Seconds between two consecutive loaded frames. */
export function frameDt(file) {
  return file.src.frameInterval() * (file.window ? file.window.step : 1);
}

export async function runTracking() {
  const file = activeFile();
  if (!file || file.loaded.length < 3) { toast('Tracking needs a movie with at least 3 loaded frames.', 'warn'); return null; }
  stopPlayback();
  const chans = analysisChannels(file);
  const job = { cancelled: false };
  const total = file.loaded.length * chans.length;
  S.busy = { label: 'Tracking particles', done: 0, total, cancel: () => { job.cancelled = true; } };
  bump();
  await tick();
  const tracks = [];
  let done = 0, last = performance.now(), detections = 0;
  try {
    for (const ch of chans) {
      const rect = channelRect(file.scope, ch, file.src);
      if (!rect) continue;
      const o = { ...fitOptions(file, ch), maxPeaks: 600 };
      const perFrame = [];
      for (let i = 0; i < file.loaded.length; i++) {
        if (job.cancelled) break;
        const data = file.frames.get(file.loaded[i]);
        const img = ch === 'all' ? data : cropRect(data, file.src.width, rect);
        const fits = detectAndFit(img, rect.w, rect.h, o).fits;
        detections += fits.length;
        perFrame.push(fits.map((q) => ({ x: q.x + rect.x, y: q.y + rect.y, brightness: q.brightness, sx: q.sx, sy: q.sy, amp: q.amp })));
        done++;
        if (performance.now() - last > 50) { S.busy.done = done; last = performance.now(); bump(); await tick(); }
      }
      if (job.cancelled) break;
      for (const t of linkTracks(perFrame, S.track)) {
        t.channel = ch === 'all' ? (channelAt(file.scope, file.src, t.points[0].x, t.points[0].y) || 'all') : ch;
        tracks.push(t);
      }
    }
  } finally {
    S.busy = null;
  }
  if (job.cancelled) { bump(); return null; }
  const dt = frameDt(file);
  tracks.forEach((t, i) => { t.id = i + 1; t.stats = trackStats(t, dt, file.pixUm); });
  file.tracks = { list: tracks, dt, params: { ...S.track }, detections, frames: file.loaded.length };
  S.movie.showTracks = true;
  S.sel = null;
  bump('overlay');
  toast(tracks.length ? `${tracks.length} tracks from ${detections} localizations.` : 'No tracks found. Lower the threshold or the minimum track length.', tracks.length ? 'info' : 'warn');
  return file.tracks;
}

export function clearTracks(file = activeFile()) {
  if (!file) return;
  file.tracks = null;
  if (S.sel && S.sel.trackId) S.sel = null;
  bump('overlay');
}

/** Brightness of one spot in every loaded frame (PsyFit's per-frame fit at a fixed position). */
export function runTrace(fit, penalty = 1) {
  const file = fileById(fit.fileId);
  if (!file || file.loaded.length < 2) { toast('A time trace needs a movie.', 'warn'); return null; }
  const ch = fit.run || fit.channel;
  const rect = channelRect(file.scope, ch, file.src) || { x: 0, y: 0, w: file.src.width, h: file.src.height };
  const dt = frameDt(file);
  const t = [], y = [], ok = [];
  for (let i = 0; i < file.loaded.length; i++) {
    const data = file.frames.get(file.loaded[i]);
    const img = ch === 'all' ? data : cropRect(data, file.src.width, rect);
    const r = fitPsf(img, rect.w, rect.h, fit.lpx, fit.lpy, { pixUm: file.pixUm, refine: false, traceBounds: true, sigThresholdUm: 0.5 });
    t.push(i * dt);
    y.push(r ? r.brightness : NaN);
    ok.push(!!(r && r.ok));
  }
  S.trace = { fileId: file.id, fitId: fit.id, channel: fit.channel, x: fit.x, y: fit.y, t, values: y, ok, dt, penalty };
  updateTraceSteps(penalty);
  return S.trace;
}

export function updateTraceSteps(penalty) {
  const tr = S.trace;
  if (!tr) return;
  tr.penalty = penalty;
  const clean = tr.values.map((v) => (Number.isFinite(v) ? v : 0));
  const bounds = detectStepsKV(clean, penalty);
  const st = staircaseFromBounds(clean, bounds);
  tr.bounds = bounds; tr.levels = Array.from(st.levels); tr.steps = st.steps;
  const drops = st.steps.filter((s) => s.drop > 0).map((s) => s.drop);
  tr.unit = drops.length ? median(drops) : NaN;
  bump();
}

// ---------------------------------------------------------------------------
// Nikon spectra: the calibration, and the spectrum of each localization
// ---------------------------------------------------------------------------

/** Use a calibration (null clears it). Real ones are remembered on this device. */
export function setCalibration(cal, remember = true) {
  S.spec.cal = cal;
  if (cal && remember && !cal.demo) calStore.save(calibrationToJson(cal));
  bump('overlay');
}

/** "YYMMDD" of a file's acquisition, as calibration files are dated. */
function acquisitionDate(file) {
  const t = file.src.info.ExperimentTime;
  if (!(t > 0)) return '';
  const d = new Date(t * 1000);
  const two = (n) => String(n).padStart(2, '0');
  return `${two(d.getFullYear() % 100)}${two(d.getMonth() + 1)}${two(d.getDate())}`;
}

/** Calibrations remembered on this device: [{date, name, fitsName}] newest first. */
export function rememberedCalibrations() {
  return calStore.all().map((c) => ({ date: c.date, name: c.name, fitsName: c.fitsName }));
}

/**
 * When a Nikon file opens without a (real) calibration in use, take the
 * remembered one measured most recently before the file was acquired.
 */
function chooseCalibrationFor(file) {
  if (file.origin && file.origin.kind === 'demo') return;
  if (S.spec.cal && !S.spec.cal.demo) return;
  const saved = calStore.all();
  const date = acquisitionDate(file);
  const pick = saved.find((c) => !date || !c.date || c.date <= date) || saved[saved.length - 1] || null;
  S.spec.cal = pick ? calibrationFromJson(pick) : null;
}

export function useRememberedCalibration(date, name) {
  const hit = calStore.all().find((c) => c.date === date && c.name === name);
  const cal = hit ? calibrationFromJson(hit) : null;
  if (cal) { setCalibration(cal, false); toast(`Using the calibration of ${cal.date || cal.name}.`); }
}

/** Open the two calibration .pkl files chosen with the system picker. */
export async function openCalibrationFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return false;
  try {
    const big = files.find((f) => f.size > CAL_MAX_BYTES);
    if (big) throw new Error(`${big.name} is ${(big.size / 1048576).toFixed(0)} MB: that is the full recording. Choose the small "..._no_tracking.pkl" file instead.`);
    const read = await Promise.all(files.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
    const cal = readCalibrationFiles(read);
    setCalibration(cal);
    toast(`Calibration ${cal.date || cal.name} loaded: ${cal.n} points, ${cal.withFit} with a wavelength fit.`);
    return true;
  } catch (err) {
    toast(err && err.message ? err.message : 'Those files could not be read as a calibration.', 'error');
    return false;
  }
}

/** List the calibrations in Drive (S.spec.found). */
export async function findDriveCalibrations() {
  S.spec.busy = 'Looking in Drive'; S.spec.found = null; bump();
  try {
    S.spec.found = await drive.calibrations();
    if (!S.spec.found.length) toast('No calibration files were found in Drive (a "saving_info_..._no_tracking.pkl" with its "..._fits.pkl").', 'warn', 5000);
  } catch (err) {
    toast(driveErrorText(err), 'error');
  }
  S.spec.busy = ''; bump();
  return S.spec.found;
}

/** Fetch one of the calibrations listed by findDriveCalibrations and use it. */
export async function loadDriveCalibration(entry) {
  S.spec.busy = `Fetching the calibration of ${entry.date}`; bump();
  try {
    const grid = await drive.downloadRaw(entry.grid);
    const fits = await drive.downloadRaw(entry.fits);
    const cal = readCalibrationFiles([{ name: entry.grid.name, bytes: grid }, { name: entry.fits.name, bytes: fits }]);
    S.spec.busy = '';
    setCalibration(cal);
    toast(`Calibration ${cal.date} loaded: ${cal.n} points, ${cal.withFit} with a wavelength fit.`);
    return true;
  } catch (err) {
    S.spec.busy = '';
    toast(err && err.code ? driveErrorText(err) : (err && err.message) || 'The calibration could not be read.', 'error');
    return false;
  }
}

/**
 * One step from "no calibration" to a spectrum: look in Drive and take the
 * calibration measured most recently before the file was acquired. Without
 * Drive (or without a match) the Fit panel opens, where files can be chosen.
 */
export async function autoCalibrate(file = activeFile()) {
  const toPanel = () => { if (S.sheet !== 'analyze') setSheet('analyze'); return false; };
  if (!(await drive.available())) return toPanel();
  const found = await findDriveCalibrations();
  if (!found || !found.length) return toPanel();
  const date = file ? acquisitionDate(file) : '';
  return loadDriveCalibration(found.find((c) => !date || c.date <= date) || found[found.length - 1]);
}

/** Every usable localization of a Nikon file in the image shown now (automatic fits, then double-tapped ones). */
function nikonLocalizations(file) {
  const auto = file.fitsKey === frameKey(file) ? file.fits.filter((q) => q.ok && q.channel === 'loc') : [];
  const manual = file.manual.filter((q) => q.channel === 'loc' && q.frame === frameKey(file) && !auto.some((a) => Math.hypot(a.x - q.x, a.y - q.y) < 1.5));
  return auto.concat(manual);
}

/** Frames a spectrum is averaged over: the one shown, or five around it in a movie. */
function spectrumFrames(file) {
  if (!file.isMovie || file.view !== 'frame' || file.loaded.length < 2) return [frameData(file)];
  const n = file.loaded.length, half = 2;
  const first = Math.max(0, Math.min(n - (2 * half + 1), file.cur - half));
  const out = [];
  for (let i = first; i < Math.min(n, first + 2 * half + 1); i++) out.push(file.frames.get(file.loaded[i]));
  return out;
}

/**
 * The spectrum that belongs to a localization in a Nikon frame.
 * @returns null for anything that has no spectrum (another microscope or
 *   region), else {status} with status 'no-cal' | 'unmapped' | 'edge' | 'ok';
 *   'ok' adds nm, values (pps, averaged over 5 rows, and over 5 frames in a
 *   movie), net (minus their minimum), xs/ys, win (sampling window on the
 *   sensor), frames, leftEdge and overlaps (other localizations whose spectra
 *   may cross this one, by PsyFit's rule).
 */
export function spectrumFor(fit) {
  const file = fit && fileById(fit.fileId);
  if (!file || file.scope !== 'nikon' || fit.channel !== 'loc') return null;
  const cal = S.spec.cal;
  if (!cal) return { status: 'no-cal' };
  const W = file.src.width, H = file.src.height;
  const key = `${fit.id}:${frameKey(file)}:${cal.date}:${cal.name}:${file.fits.length}:${file.manual.length}`;
  if (!file.specCache) file.specCache = new Map();
  if (file.specCache.has(key)) return file.specCache.get(key);
  const compute = () => {
    const [ox, oy] = objectPoint(fit.x, fit.y, H);
    const pos = mapToSpectrum(cal, ox, oy);
    if (!pos) return { status: 'unmapped' };
    const frames = spectrumFrames(file);
    const values = new Float64Array(SPEC_NPTS);
    for (const fr of frames) {
      const one = fr && sampleSpectrum(fr, W, H, pos[0], pos[1]);
      if (!one) return { status: 'edge' };
      for (let i = 0; i < SPEC_NPTS; i++) values[i] += one[i] / frames.length;
    }
    const axis = wavelengthAxis(cal, ox, oy);
    if (!axis) return { status: 'unmapped' };
    let min = Infinity;
    for (let i = 0; i < SPEC_NPTS; i++) if (values[i] < min) min = values[i];
    const overlaps = [];
    for (const q of nikonLocalizations(file)) {
      if (q.id === fit.id || Math.hypot(q.x - fit.x, q.y - fit.y) < 1.5) continue;
      const other = mapToSpectrum(cal, ...objectPoint(q.x, q.y, H));
      if (other && spectraOverlap(pos, other)) overlaps.push({ id: q.id, win: spectralWindow(other[0], other[1], H) });
    }
    return {
      status: 'ok', xs: pos[0], ys: pos[1], nm: axis.nm, nmPerSample: axis.nmPerSample, values, net: values.map((v) => v - min),
      win: spectralWindow(pos[0], pos[1], H), frames: frames.length, leftEdge: offDetector(pos[0]), overlaps,
    };
  };
  const out = compute();
  if (file.specCache.size > 400) file.specCache.clear();
  file.specCache.set(key, out);
  return out;
}

/** Spectra of every localization of a Nikon file: [{fit, sp}] for those that have one. */
export function spectraOf(file = activeFile()) {
  if (!file || file.scope !== 'nikon' || !S.spec.cal) return [];
  const out = [];
  for (const fit of nikonLocalizations(file)) {
    const sp = spectrumFor(fit);
    if (sp && sp.status === 'ok') out.push({ fit, sp });
  }
  return out;
}

/** Spectra as a table PsyFit's Process Spectra reads (Get Spectra's columns, plus two flags). */
export function spectraCsv(file = activeFile()) {
  const cal = S.spec.cal;
  const rows = [];
  spectraOf(file).forEach(({ fit, sp }, n) => {
    const [ox, oy] = objectPoint(fit.x, fit.y, file.src.height);
    for (let i = 0; i < SPEC_NPTS; i++) {
      rows.push([file.name, n, g6(sp.nm[i]), g6(sp.net[i]), g6(ox), g6(oy), g6(fit.brightness), sp.overlaps.length ? 1 : 0, sp.leftEdge ? 1 : 0, cal.name, cal.fitsName]);
    }
  });
  return csv(['File', 'Particle_ID', 'Wavelength_nm', 'Intensity', 'Particle_X', 'Particle_Y', 'Brightness_pps', 'Possible_Overlap', 'Off_Detector', 'Calibration_File', 'Calibration_Fits'], rows);
}

// ---------------------------------------------------------------------------
// Tables and exports
// ---------------------------------------------------------------------------

const csvCell = (v) => {
  if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '';
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
const csv = (header, rows) => [header.join(','), ...rows.map((r) => r.map(csvCell).join(','))].join('\n') + '\n';
const g6 = (v) => (Number.isFinite(v) ? Number(v.toPrecision(7)) : v);

/** PsyFit-compatible fit table (same column names and units), plus a few extras. */
export function fitsCsv(pool = 'file') {
  const files = pool === 'all' ? S.files : [activeFile()].filter(Boolean);
  const header = ['x_pix', 'y_pix', 'x_um', 'y_um', 'amp_fit', 'sigx_fit', 'sigy_fit', 'brightness_fit', 'brightness_integrated',
    'gainDAC', 'exposure_time_sec', 'accumulated_cycles', 'file_stem', 'quadrant', 'frame', 'x_sensor', 'y_sensor',
    'offset_fit', 'snr', 'r2', 'loc_precision_nm', 'pix_size_um', 'manual'];
  const rows = [];
  for (const f of files) {
    const info = f.src.info;
    const list = (f.fitsKey === frameKey(f) ? f.fits : []).concat(f.manual);
    for (const q of list) {
      const p = f.pixUm;
      rows.push([q.lpx, q.lpy, g6(q.lx * p), g6(q.ly * p), g6(q.amp), g6(q.sx * p), g6(q.sy * p), g6(q.brightnessFit), g6(q.brightness),
        info.GainDAC > 0 ? info.GainDAC : 1, info.ExposureTime, info.AccumulatedCycles, f.name.replace(/\.sif$/i, ''),
        channelInfo(f.scope, q.channel).label === 'All' ? q.channel : channelInfo(f.scope, q.channel).label,
        q.manual ? q.frame : frameKey(f), g6(q.x), g6(q.y), g6(q.offset), g6(q.snr), g6(q.r2), g6(locPrecisionNm(q, f)), p, q.manual ? 1 : 0]);
    }
  }
  return csv(header, rows);
}

export function tracksCsv(file = activeFile()) {
  if (!file || !file.tracks) return '';
  const rows = [];
  for (const t of file.tracks.list) {
    for (const p of t.points) {
      rows.push([t.id, t.channel, file.loaded[p.f] + 1, g6(p.f * file.tracks.dt), g6(p.x), g6(p.y), g6(p.x * file.pixUm), g6(p.y * file.pixUm),
        g6(p.brightness), g6(p.sx * file.pixUm), g6(p.sy * file.pixUm), g6(t.stats.D), g6(t.stats.alpha)]);
    }
  }
  return csv(['track', 'channel', 'frame', 'time_s', 'x_pix', 'y_pix', 'x_um', 'y_um', 'brightness_integrated', 'sigx_um', 'sigy_um', 'D_um2_s', 'alpha'], rows);
}

export function traceCsv() {
  const tr = S.trace;
  if (!tr) return '';
  const file = fileById(tr.fileId);
  return csv(['frame', 'time_s', 'brightness_integrated', 'fit_ok', 'step_level'],
    tr.values.map((v, i) => [file ? file.loaded[i] + 1 : i + 1, g6(tr.t[i]), g6(v), tr.ok[i] ? 1 : 0, g6(tr.levels[i])]));
}

export function colocCsv() {
  const r = S.coloc.result;
  if (!r) return '';
  return csv(['a_x', 'a_y', 'a_brightness', 'b_x', 'b_y', 'b_brightness', 'distance_px'],
    r.pairs.map((p) => [g6(p.a.x), g6(p.a.y), g6(p.a.brightness), g6(p.b.x), g6(p.b.y), g6(p.b.brightness), g6(p.d)]));
}

export function setSheet(name) {
  S.sheet = S.sheet === name ? null : name;
  bump();
}

export function select(sel) {
  S.sel = sel;
  bump('overlay');
}

/** Open or close the colour-scale editor (the card that sets min and max). */
export function setScaleEditor(open) {
  if (S.scale === !!open) return;
  S.scale = !!open;
  bump('overlay');
}

export { HUE_NAMES, gaussianFilter };
