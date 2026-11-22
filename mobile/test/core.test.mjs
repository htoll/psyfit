// Core checks that run in plain Node:  npm test   (or: node --test)
//
// They cover the file reader, the microscope layouts, the colour mapping and,
// on the simulated demo data (whose ground truth is known), detection,
// fitting, channel registration and tracking.

import test from 'node:test';
import assert from 'node:assert/strict';

import { writeSif, openSif, parseSifHeader, ppsScale } from '../src/core/sif.js';
import {
  SCOPES, channelRect, availableChannels, channelAt, channelInfo, cropRect, guessScope, pixelSize, NIKON_BEAM_PX,
} from '../src/core/layouts.js';
import { COLORMAPS, HUE_NAMES, lutRgb, makeNorm, composeScene } from '../src/core/render.js';
import { detectAndFit } from '../src/core/numeric.js';
import { estimateRegistration, applyAffine, invertAffine } from '../src/core/geometry.js';
import { linkTracks, trackStats } from '../src/core/tracking.js';
import { makeDemoImage, makeDemoMovie, DEMO_CHANNEL_AFFINES } from '../src/core/demo.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || 'value'}: ${a} vs ${b} (tol ${tol})`);

test('sif: written files read back exactly, in every supported header layout', async () => {
  for (const version of [65555, 65558, 65559, 65565, 65567]) {
    const w = 12, h = 7;
    const frames = [0, 1, 2].map((f) => Float32Array.from({ length: w * h }, (_, i) => f * 1000 + i + 0.5));
    const bytes = writeSif({ width: w, height: h, frames, exposure: 0.05, cycle: 0.06, gain: 50, accumulations: 10, version });
    const info = parseSifHeader(bytes);
    assert.equal(info.SifVersion, version);
    assert.deepEqual([info.width, info.height, info.frames], [w, h, 3]);
    close(info.ExposureTime, 0.05, 1e-12, 'exposure');
    assert.equal(info.GainDAC, 50);
    assert.equal(info.AccumulatedCycles, 10);
    const src = await openSif(new Blob([bytes]), 'roundtrip.sif');
    assert.equal(src.frames, 3);
    assert.equal(src.truncated, false);
    for (let f = 0; f < 3; f++) assert.deepEqual(Array.from(await src.readFrame(f)), Array.from(frames[f]), `frame ${f}, version ${version}`);
    close(ppsScale(info), 5 / 50 / 0.05 / 10, 1e-12, 'counts to pps');
  }
});

test('sif: a file cut short reports the frames that are present', async () => {
  const frames = [0, 1, 2, 3].map(() => new Float32Array(64).fill(7));
  const bytes = writeSif({ width: 8, height: 8, frames });
  const cut = bytes.subarray(0, bytes.length - 2 - 8 * 8 * 4 - 10);      // lose the last frame and a bit more
  const src = await openSif(new Blob([cut]), 'cut.sif');
  assert.equal(src.frames, 2);
  assert.equal(src.truncated, true);
  await assert.rejects(() => openSif(new Blob([new Uint8Array(200)]), 'zeros.sif'), /Not an Andor/);
});

test('layouts: MCL quadrants are the array slices PsyFit uses', () => {
  const src = { width: 512, height: 512 };
  // region 1 Blue = [256:512, 0:256], 2 Green = [256:512, 256:512], 3 Red = [0:256, 0:256], 4 NIR = [0:256, 256:512]
  assert.deepEqual(channelRect('mcl', 'blue', src), { x: 0, y: 256, w: 256, h: 256 });
  assert.deepEqual(channelRect('mcl', 'green', src), { x: 256, y: 256, w: 256, h: 256 });
  assert.deepEqual(channelRect('mcl', 'red', src), { x: 0, y: 0, w: 256, h: 256 });
  assert.deepEqual(channelRect('mcl', 'nir', src), { x: 256, y: 0, w: 256, h: 256 });
  assert.deepEqual(channelRect('mcl', 'all', src), { x: 0, y: 0, w: 512, h: 512 });
  assert.equal(channelRect('mcl', 'ch1', src), null);
  assert.deepEqual(availableChannels('mcl', src).map((c) => c.id), ['blue', 'green', 'red', 'nir']);
  assert.equal(channelAt('mcl', src, 300, 10), 'nir');
  assert.equal(channelAt('mcl', src, 10, 300), 'blue');
  assert.equal(channelInfo('mcl', 'nir').label, 'NIR');
  assert.equal(channelInfo('mcl', 'all').label, 'All');
  assert.equal(channelInfo('full', 'border').label, 'Border');
  assert.deepEqual(availableChannels('full', src), []);

  const img = Float32Array.from({ length: 6 * 4 }, (_, i) => i);      // 6 wide, 4 tall
  assert.deepEqual(Array.from(cropRect(img, 6, { x: 1, y: 2, w: 3, h: 2 })), [13, 14, 15, 19, 20, 21]);
});

test('layouts: Nikon regions, scope guessing and pixel sizes', () => {
  const src = { width: 512, height: 512 };
  // Localization image: PsyFit's "Mr Beam" region image[-280:, :280] (drawn at the top left). Spectra: bottom right.
  assert.deepEqual(channelRect('nikon', 'loc', src), { x: 0, y: 512 - NIKON_BEAM_PX, w: NIKON_BEAM_PX, h: NIKON_BEAM_PX });
  assert.deepEqual(channelRect('nikon', 'spec', src), { x: 200, y: 0, w: 312, h: 280 });
  assert.deepEqual(SCOPES.nikon.channels.map((c) => c.label), ['Localization', 'Spectra']);
  assert.equal(channelAt('nikon', src, 100, 400), 'loc');
  assert.equal(channelAt('nikon', src, 400, 100), 'spec');
  assert.equal(channelAt('nikon', src, 100, 100), null);
  assert.equal(channelRect('nikon', 'blue', src), null);

  assert.equal(guessScope('Demo_simulated_MCL_976nm_4channel.sif', {}), 'mcl');
  assert.equal(guessScope('sample3.sif', { OriginalFilename: 'D:\\20261117_Nikon\\sample3\\sample3.sif' }), 'nikon');
  assert.equal(guessScope('MrBeam_fov2.sif', {}), 'nikon');
  assert.equal(guessScope('emclass_test.sif', {}), '');
  assert.equal(pixelSize('mcl', 'a.sif', {}), 0.1011);
  assert.equal(pixelSize('nikon', 'a.sif', {}), 0.107);
  close(pixelSize('nikon', 'UCNP_60x_1e4.sif', {}), 0.107 * 100 / 60, 1e-12, '60x Nikon');
  close(pixelSize('mcl', 'a.sif', { xbin: 2 }), 0.2022, 1e-12, 'binned');
});

test('render: colour scale mapping and compositing', () => {
  assert.equal(COLORMAPS[0], 'grey');
  const grey = lutRgb('grey');
  assert.deepEqual([grey[0], grey[1], grey[2], grey[765], grey[766], grey[767]], [0, 0, 0, 255, 255, 255]);
  assert.equal(lutRgb('gray'), grey);
  for (const name of [...COLORMAPS, ...HUE_NAMES]) assert.equal(lutRgb(name).length, 768, name);

  const lin = makeNorm(100, 300, false);
  assert.equal(lin.fwd(200), 0.5);
  assert.equal(lin.fwd(50), 0);
  assert.equal(lin.fwd(1e9), 1);
  assert.equal(lin.fwd(NaN), 0);
  close(lin.inv(0.25), 150, 1e-9);
  for (const [lo, hi] of [[10, 1000], [-5, 400], [0, 50]]) {
    const log = makeNorm(lo, hi, true);
    close(log.fwd(lo), 0, 1e-12); close(log.fwd(hi), 1, 1e-12);
    for (const t of [0.1, 0.5, 0.9]) close(log.fwd(log.inv(t)), t, 1e-9, `log round trip ${lo}..${hi}`);
  }
  close(makeNorm(10, 1000, true).inv(0.5), 100, 1e-9, 'geometric midpoint');

  // One layer: image row 0 is drawn at the bottom; lo is black and hi is white.
  const data = Float32Array.from([0, 10, 20, 30]);                    // 2 x 2, rows [0,10] and [20,30]
  const layer = { data, sw: 2, rect: { x: 0, y: 0, w: 2, h: 2 }, m: null, inv: null, lo: 0, hi: 30, log: false, lut: grey };
  const out = new Uint8ClampedArray(16);
  composeScene({ w: 2, h: 2, layers: [layer] }, out);
  assert.deepEqual([out[0], out[4], out[8], out[12]], [170, 255, 0, 85]);
  assert.deepEqual([out[3], out[7], out[11], out[15]], [255, 255, 255, 255]);

  // Two layers add; a shifted layer is resampled and leaves uncovered pixels alone.
  const flat = (v) => ({ data: new Float32Array(16).fill(v), sw: 4, rect: { x: 0, y: 0, w: 4, h: 4 }, inv: null, lo: 0, hi: 1, log: false });
  const red = { ...flat(1), lut: lutRgb('red') };
  const shift = [1, 0, 2, 0, 1, 0];                                    // 2 px to the right
  const green = { ...flat(1), lut: lutRgb('green'), m: shift, inv: invertAffine(shift) };
  const rgba = new Uint8ClampedArray(64);
  composeScene({ w: 4, h: 4, layers: [red, green] }, rgba);
  assert.deepEqual(Array.from(rgba.subarray(0, 4)), [255, 42, 0, 255]);            // x = 0: red only
  assert.deepEqual(Array.from(rgba.subarray(12, 16)), [255, 255, 133, 255]);       // x = 3: both, clipped at 255
});

test('demo image: every particle is found in the bright channels and brightness tracks the truth', () => {
  const demo = makeDemoImage();
  const info = parseSifHeader(demo.bytes);
  const raw = new Float32Array(demo.bytes.buffer.slice(info.offset, info.offset + info.frameBytes));
  const k = ppsScale(info);
  const pps = raw.map((v) => v * k);
  const src = { width: info.width, height: info.height };
  const found = {};
  for (const ch of ['blue', 'green', 'red']) {
    const rect = channelRect('mcl', ch, src);
    const { fits } = detectAndFit(cropRect(pps, src.width, rect), rect.w, rect.h, { threshold: 4, minDistance: 5, pixUm: 0.1011 });
    found[ch] = fits;
    assert.equal(fits.length, demo.truth.particles.length, `${ch}: particles found`);
  }
  // In green (the reference frame) each fit sits on a true particle, and a 5-mer is ~5x a monomer.
  const bright = {};
  for (const p of demo.truth.particles) {
    const hit = found.green.find((q) => Math.hypot(q.x - p.x, q.y - p.y) < 1.0);
    assert.ok(hit, `particle at ${p.x.toFixed(1)}, ${p.y.toFixed(1)} located within a pixel`);
    (bright[p.order] = bright[p.order] || []).push(hit.brightness / p.jitter);
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const mono = mean(bright[1]);
  for (const order of Object.keys(bright)) close(mean(bright[order]) / mono, Number(order), 0.12 * order, `brightness of ${order}-mers over monomers`);

  // Registration onto green recovers the simulated misalignment of the other channels.
  const pts = (ch) => found[ch].map((q) => [q.x, q.y]);
  for (const ch of ['blue', 'red']) {
    const r = estimateRegistration([{ src: pts(ch), dst: pts('green') }], { maxShift: 40 });
    assert.ok(r && r.kind === 'affine', `${ch} registered`);
    const want = DEMO_CHANNEL_AFFINES[ch];
    for (const [x, y] of [[20, 20], [128, 128], [230, 40]]) {
      const a = applyAffine(r.m, x, y), b = applyAffine(want, x, y);
      close(Math.hypot(a[0] - b[0], a[1] - b[1]), 0, 0.25, `${ch} mapping at ${x}, ${y}`);
    }
  }
});

test('demo movie: particles are tracked and their diffusion coefficients are in range', () => {
  const demo = makeDemoMovie(5, 40);
  const info = parseSifHeader(demo.bytes);
  const k = ppsScale(info);
  const perFrame = [];
  for (let f = 0; f < info.frames; f++) {
    const start = info.offset + f * info.frameBytes;
    const frame = new Float32Array(demo.bytes.buffer.slice(start, start + info.frameBytes)).map((v) => v * k);
    const { fits } = detectAndFit(frame, info.width, info.height, { threshold: 10, minDistance: 5, pixUm: demo.truth.pix, maxPeaks: 600 });
    perFrame.push(fits.map((q) => ({ x: q.x, y: q.y, brightness: q.brightness })));
  }
  const tracks = linkTracks(perFrame, { maxDisp: 5, maxGap: 2, minLength: 5 });
  assert.ok(tracks.length >= 12 && tracks.length <= 30, `${tracks.length} tracks`);
  // Each simulated mover's first position starts a track; its D is recovered to within a factor of ~2.
  let matched = 0;
  for (const mover of demo.truth.movers) {
    const [x0, y0] = mover.path[0];
    const t = tracks.find((tr) => tr.points[0].f === 0 && Math.hypot(tr.points[0].x - x0, tr.points[0].y - y0) < 1.5);
    if (!t || t.points.length < 25) continue;
    const D = trackStats(t, demo.truth.dt, demo.truth.pix).D;
    assert.ok(D > mover.D / 2.5 && D < mover.D * 2.5, `D ${D.toFixed(3)} vs true ${mover.D.toFixed(3)}`);
    matched++;
  }
  assert.ok(matched >= 8, `${matched} movers followed for 25+ frames`);
});
