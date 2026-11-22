// Nikon spectra: the calibration reader, the localization-to-spectrum mapping,
// the sampling of a spectrum and the overlap test.   npm test
//
// The .pkl fixtures are synthetic (see fixtures/make_fixtures.py). The same
// code was checked against PsyFit's Get Spectra on a real Nikon frame with its
// real calibration: identical particles, spectra and overlap flags.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { unpickle, PickleError } from '../src/core/pickle.js';
import {
  readCalibrationFiles, calibrationKind, calibrationDate, calibrationToJson, calibrationFromJson, CalibrationError,
  mapToSpectrum, wavelengthAxis, sampleSpectrum, objectPoint, spectraOverlap, offDetector, spectralWindow,
  SPEC_LEFT, SPEC_RIGHT, SPEC_NPTS,
} from '../src/core/spectra.js';
import { parseSifHeader, ppsScale } from '../src/core/sif.js';
import { channelRect, cropRect } from '../src/core/layouts.js';
import { detectAndFit } from '../src/core/numeric.js';
import { makeDemoNikon } from '../src/core/demo.js';

const fixture = (name) => new Uint8Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
const GRID = 'saving_info_990101_no_tracking.pkl', FITS = '990101_fits.pkl';
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || 'value'}: ${a} vs ${b} (tol ${tol})`);

test('pickle: reads NumPy-written dicts of tuples, scalars and arrays, and nothing else', () => {
  const grid = unpickle(fixture(GRID));
  assert.ok(grid instanceof Map);
  assert.equal(grid.size, 24);
  const p7 = grid.get('P0007');                       // i = 1, j = 1
  assert.equal(p7[0], 'sim');
  assert.deepEqual(p7[1], [35, 40]);
  assert.ok(p7[2] instanceof Float64Array);
  assert.deepEqual(Array.from(p7[2]), [30007, 35, 40, 3, 2.5, 0, 24000]);
  close(p7[3][0], 1.03 * 35 + 190, 1e-12); close(p7[3][1], 1.01 * 40 + 250, 1e-12);
  assert.deepEqual([p7[4].length, p7[4][0], p7[4][99], p7[4].shape[0]], [100, 7, 106, 100]);
  const fits = unpickle(fixture(FITS));
  assert.equal(fits.size, 12);
  assert.deepEqual(fits.get('P0002'), [56.2, 0.022, 376]);

  assert.throws(() => unpickle(new TextEncoder().encode('Andor Technology Multi-Channel File\n')), PickleError);
  assert.throws(() => unpickle(fixture(GRID).subarray(0, 500)), /ends in the middle/);
  // a pickle that would call os.system on load is refused, not run
  const evil = Uint8Array.from([0x80, 0x02, ...new TextEncoder().encode('cos\nsystem\n'), 0x58, 2, 0, 0, 0, 0x6c, 0x73, 0x85, 0x52, 0x2e]);
  assert.throws(() => unpickle(evil), /does not read \(os\.system\)/);
});

test('calibration: the two files are told apart by name, checked, and survive storage', () => {
  assert.equal(calibrationKind('saving_info_261116_no_tracking.pkl'), 'grid');
  assert.equal(calibrationKind('261116_fits_456_474.pkl'), 'fits');
  assert.equal(calibrationKind('notes.pkl'), null);
  assert.equal(calibrationDate('saving_info_261116_no_tracking.pkl'), '261116');
  assert.equal(calibrationDate('261116_fits.pkl'), '261116');

  const files = [{ name: FITS, bytes: fixture(FITS) }, { name: GRID, bytes: fixture(GRID) }];       // any order
  const cal = readCalibrationFiles(files);
  assert.deepEqual([cal.n, cal.withFit, cal.date, cal.name, cal.fitsName], [24, 12, '990101', GRID, FITS]);
  assert.deepEqual([cal.obj[0], cal.obj[1], cal.amp[0]], [20, 30, 30000]);
  const back = calibrationFromJson(JSON.parse(JSON.stringify(calibrationToJson(cal))));
  assert.deepEqual(Array.from(back.obj), Array.from(cal.obj));
  assert.deepEqual(Array.from(back.img), Array.from(cal.img));
  assert.deepEqual(Array.from(back.fit).map(String), Array.from(cal.fit).map(String));              // NaN where there is no fit
  assert.equal(calibrationFromJson({ v: 2 }), null);

  assert.throws(() => readCalibrationFiles([files[1]]), CalibrationError);
  assert.throws(() => readCalibrationFiles([{ name: GRID, bytes: fixture(FITS) }, files[0]]), CalibrationError);   // fits content in the grid's place
  assert.throws(() => readCalibrationFiles([{ name: GRID, bytes: new Uint8Array(40) }, files[0]]), /saving_info.*Not a Python pickle/);
});

test('mapping: a particle is carried to its spectrum through the nearest calibration points', () => {
  const cal = readCalibrationFiles([{ name: GRID, bytes: fixture(GRID) }, { name: FITS, bytes: fixture(FITS) }]);
  // The fixture's map is exactly affine, so local interpolation must reproduce it anywhere.
  for (const [x, y] of [[20, 30], [47.3, 41.9], [88.8, 55.5], [12, 25], [100, 65]]) {
    const [xs, ys] = mapToSpectrum(cal, x, y);
    close(xs, 1.03 * x + 190, 1e-9, `xs at ${x}, ${y}`);
    close(ys, 1.01 * y + 250, 1e-9, `ys at ${x}, ${y}`);
  }
  assert.deepEqual(objectPoint(100.5, 400.25, 512), [100.5, 111.75]);

  // Wavelengths come from the nearest point that has a fit: (35, 40) is P0007 (no fit), its neighbours P0006 and P0008 have one.
  const axis = wavelengthAxis(cal, 36, 40);
  assert.equal(axis.id, 'P0008');                     // i = 2 at x = 50 is nearer than i = 0 at x = 20
  assert.equal(axis.nm.length, SPEC_NPTS);
  close(axis.nm[0], 56.2 + 376, 1e-12); close(axis.nm[50], 56.2 * Math.exp(1.1) + 376, 1e-9);
  close(axis.nmPerSample[10], 56.2 * (Math.exp(0.022 * 10.5) - Math.exp(0.022 * 9.5)), 1e-9);
});

test('sampling: five rows of the flipped frame, interpolated along the streak', () => {
  const W = 512, H = 512;
  // value = column + 1000 * flipped row, so the answer can be written down
  const data = new Float32Array(W * H);
  for (let row = 0; row < H; row++) for (let x = 0; x < W; x++) data[row * W + x] = x + 1000 * (H - 1 - row);
  const s = sampleSpectrum(data, W, H, 300.25, 400.4);
  assert.equal(s.length, SPEC_NPTS);
  close(s[0], 300.25 - SPEC_LEFT + 1000 * 400, 1e-6, 'first sample: mean of rows 398..402');
  close(s[99], 300.25 + SPEC_RIGHT + 1000 * 400, 1e-6, 'last sample');
  close(s[40] - s[39], 1, 1e-6, 'one pixel per sample');
  // off the left edge the row's first value is repeated (numpy.interp), and the spectrum is flagged
  const left = sampleSpectrum(data, W, H, 50, 300);
  close(left[0], 0 + 1000 * 300, 1e-6); close(left[SPEC_LEFT - 50], 1000 * 300, 1e-6); close(left[99], 59 + 1000 * 300, 1e-6);
  assert.equal(offDetector(50), true);
  assert.equal(offDetector(90), false);
  // the frame's last rows: fewer rows are averaged; beyond them there is no spectrum
  close(sampleSpectrum(data, W, H, 300, 510.2)[SPEC_LEFT], 300 + 1000 * 509.5, 1e-6, 'rows 508..511');
  close(sampleSpectrum(data, W, H, 300, 511)[SPEC_LEFT], 300 + 1000 * 510, 1e-6, 'rows 509..511');
  assert.equal(sampleSpectrum(data, W, H, 300, 1.4), null);
  assert.equal(sampleSpectrum(data, W, H, 300, 512), null);
  // 0.5 rounds to the even row, as Python does
  close(sampleSpectrum(data, W, H, 300, 400.5)[SPEC_LEFT], 300 + 1000 * 400, 1e-6);
  close(sampleSpectrum(data, W, H, 300, 401.5)[SPEC_LEFT], 300 + 1000 * 402, 1e-6);

  assert.deepEqual(spectralWindow(300, 400.4, 512), { x0: 210, x1: 309, y0: 109, y1: 113 });
  assert.equal(spectraOverlap([300, 400], [398.9, 404]), true);
  assert.equal(spectraOverlap([300, 400], [399, 400]), false);          // a full window apart
  assert.equal(spectraOverlap([300, 400], [300, 404.1]), false);        // no shared rows
});

test('demo Nikon frame: every located particle shows its emission lines, and the planted overlap is flagged', () => {
  const demo = makeDemoNikon();
  const info = parseSifHeader(demo.bytes);
  const k = ppsScale(info);
  const pps = new Float32Array(demo.bytes.buffer.slice(info.offset, info.offset + info.frameBytes)).map((v) => v * k);
  const W = info.width, H = info.height, src = { width: W, height: H };
  const loc = channelRect('nikon', 'loc', src), spec = channelRect('nikon', 'spec', src);
  assert.deepEqual(loc, { x: 0, y: 232, w: 280, h: 280 });               // PsyFit's image[-280:, :280]
  assert.deepEqual(spec, { x: 200, y: 0, w: 312, h: 280 });
  const cal = demo.calibration;
  assert.ok(cal.demo && cal.n > 200 && cal.withFit > 100);

  const { fits } = detectAndFit(cropRect(pps, W, loc), loc.w, loc.h, { threshold: 4, minDistance: 5, pixUm: 0.107 });
  assert.ok(fits.length >= 30, `${fits.length} of ${demo.truth.particles.length} particles located`);
  const { lines, kinds, sampleOf } = demo.truth;
  const spots = [];
  for (const q of fits) {
    const [ox, oy] = objectPoint(q.x + loc.x, q.y + loc.y, H);
    const truth = demo.truth.particles.find((p) => Math.hypot(p.x - ox, p.y - oy) < 1.5);
    assert.ok(truth, `fit at ${ox.toFixed(1)}, ${oy.toFixed(1)} is a simulated particle`);
    const pos = mapToSpectrum(cal, ox, oy);
    close(Math.hypot(pos[0] - truth.xs, pos[1] - truth.ys), 0, 1.5, 'spectrum position (the simulated map is slightly curved)');
    spots.push({ truth, pos });
  }
  const planted = spots.filter((s) => s.truth === demo.truth.particles[0] || s.truth === demo.truth.particles[1]);
  assert.equal(planted.length, 2);
  assert.equal(spectraOverlap(planted[0].pos, planted[1].pos), true);
  let checked = 0;
  for (const s of spots) {
    if (spots.some((o) => o !== s && spectraOverlap(s.pos, o.pos))) continue;       // only clean spectra have a known shape
    const sp = sampleSpectrum(pps, W, H, s.pos[0], s.pos[1]);
    let peak = 0;
    for (let i = 1; i < SPEC_NPTS; i++) if (sp[i] > sp[peak]) peak = i;
    const strongest = lines[kinds[s.truth.kind].indexOf(1)];
    close(peak, sampleOf(strongest), 2.5, `brightest line of a kind-${s.truth.kind} particle`);
    const axis = wavelengthAxis(cal, s.truth.x, s.truth.y);
    close(axis.nm[peak], strongest, 12, 'its wavelength');
    checked++;
  }
  assert.ok(checked >= 5, `${checked} clean spectra checked`);
});
