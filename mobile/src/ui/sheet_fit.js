// Fit panel: detection settings, per-channel brightness summary, histograms,
// aggregation classes, cross-channel comparison, the fit table and, for the
// Nikon, the spectral calibration and every particle's spectrum.

import { useEffect, useRef, useState } from 'preact/hooks';
import { html, Btn, Seg, Slider, Stepper, Toggle, Section, Num, Icon } from './widgets.js';
import { Chart, Legend } from './plots.js';
import { THEME, fmtBrightness, fmtNum } from './theme.js';
import {
  S, bump, persist, toast, activeFile, setFitParam, runDetection, clearFits, brightnessSets, summarize, classify,
  CLASS_NAMES, CLASS_COLORS, fitsCsv, frameKey, select, fitsForChannel, buildScene,
  spectraOf, spectraCsv, rememberedCalibrations, useRememberedCalibration, openCalibrationFiles, findDriveCalibrations, loadDriveCalibration,
} from '../store.js';
import { autoBinCount, histogram, gmmPdf, linregress } from '../core/stats.js';
import { matchPoints, applyAffine } from '../core/geometry.js';
import { channelInfo, channelRect } from '../core/layouts.js';
import { saveFile, drive } from '../platform.js';

export const SPECTRUM_COLOR = '#FFB84D';

/** Mean of several spectra on a common wavelength grid (each has its own, slightly different, axis). */
function meanSpectrum(rows, points = 110) {
  if (!rows.length) return [];
  let lo = -Infinity, hi = Infinity;
  for (const { sp } of rows) { lo = Math.max(lo, sp.nm[0]); hi = Math.min(hi, sp.nm[sp.nm.length - 1]); }
  if (!(hi > lo)) return [];
  const out = [];
  for (let k = 0; k < points; k++) {
    const x = lo + (hi - lo) * k / (points - 1);
    let sum = 0;
    for (const { sp } of rows) {
      let i = 1;
      while (i < sp.nm.length - 1 && sp.nm[i] < x) i++;
      const t = (x - sp.nm[i - 1]) / (sp.nm[i] - sp.nm[i - 1]);
      sum += sp.net[i - 1] + t * (sp.net[i] - sp.net[i - 1]);
    }
    out.push({ x, y: sum / rows.length });
  }
  return out;
}

/** Nikon only: which calibration is in use, where to get one, and the spectra of the fitted particles. */
function SpectraSection({ file }) {
  const cal = S.spec.cal;
  const inputRef = useRef(null);
  const [driveOk, setDriveOk] = useState(null);
  useEffect(() => { let alive = true; drive.available().then((ok) => { if (alive) setDriveOk(ok); }); return () => { alive = false; }; }, []);
  const remembered = rememberedCalibrations().filter((c) => !cal || cal.demo || c.date !== cal.date || c.name !== cal.name);
  const rows = cal ? spectraOf(file) : [];
  const flagged = rows.filter((r) => r.sp.overlaps.length || r.sp.leftEdge);
  const clean = rows.filter((r) => !r.sp.overlaps.length && !r.sp.leftEdge);
  const shown = (S.spec.showOverlaps ? rows : clean).slice(0, 80);
  const exportCsv = async () => {
    try {
      const r = await saveFile(`${file.name.replace(/\.sif$/i, '')}_spectra.csv`, spectraCsv(file), 'text/csv');
      if (r !== 'declined') toast('Spectra saved as CSV.');
    } catch (err) { toast(err.message, 'error'); }
  };
  const series = shown.map(({ fit, sp }) => ({
    name: `Particle at ${fit.lx.toFixed(0)}, ${fit.ly.toFixed(0)}`, hit: false, width: 1,
    color: sp.overlaps.length || sp.leftEdge ? 'rgba(240,195,106,0.35)' : 'rgba(255,184,77,0.4)',
    points: Array.from(sp.nm, (x, i) => ({ x, y: sp.net[i] })),
  }));
  const mean = meanSpectrum(clean);
  if (mean.length) series.push({ name: 'Mean', color: THEME.ink, width: 2, points: mean });
  return html`<${Section} title="Spectra" id="spectra-section" aside=${rows.length ? html`<${Btn} small=${true} icon="save" onClick=${exportCsv} id="export-spectra">CSV<//>` : null}>
    <p class="hint" id="cal-status">${cal
      ? `${cal.demo ? 'Simulated demo calibration' : `Calibration ${cal.date || cal.name}`}: ${cal.n} grid points, ${cal.withFit} with a wavelength fit${cal.demo ? '' : ` (${cal.name})`}.`
      : 'No calibration loaded. A particle’s spectrum is found through the calibration measured for that date: the “saving_info_…_no_tracking.pkl” file and its “…_fits.pkl”.'}</p>
    <div class="btn-row">
      ${driveOk ? html`<${Btn} icon="drive" onClick=${() => findDriveCalibrations()} id="cal-drive" disabled=${!!S.spec.busy}>Find in Drive<//>` : null}
      <${Btn} icon="files" onClick=${() => inputRef.current && inputRef.current.click()} id="cal-files">Open the two .pkl files<//>
      <input ref=${inputRef} class="visually-hidden" id="cal-input" type="file" multiple tabindex="-1" aria-hidden="true"
        onChange=${async (e) => { const input = e.currentTarget; await openCalibrationFiles(input.files); input.value = ''; }} />
    </div>
    ${S.spec.busy ? html`<p class="hint" role="status">${S.spec.busy}…</p>` : null}
    ${S.spec.found && S.spec.found.length ? html`<ul class="list" id="cal-found">
      ${S.spec.found.slice(0, 12).map((c) => html`<li key=${c.date} class=${`row${cal && !cal.demo && cal.date === c.date ? ' is-on' : ''}`}>
        <button type="button" class="row-main" disabled=${!!S.spec.busy} onClick=${() => loadDriveCalibration(c)}>
          <${Icon} name="drive" size=${18} />
          <span class="row-text"><span class="row-title">Calibration ${c.date}</span><span class="row-meta">${c.grid.name} · ${c.fits.name}</span></span>
        </button></li>`)}
    </ul>` : null}
    ${remembered.length ? html`<div class="pill-row" role="group" aria-label="Calibrations on this device">
      ${remembered.map((c) => html`<button type="button" key=${`${c.date}${c.name}`} class="pill" onClick=${() => useRememberedCalibration(c.date, c.name)}>On this device: ${c.date || c.name}</button>`)}
    </div>` : null}
    ${cal && !rows.length ? html`<p class="hint">Fit the particles above (or double-tap one) to read their spectra.</p>` : null}
    ${rows.length ? html`<figure class="plot">
      <figcaption><span class="dot" style=${{ background: SPECTRUM_COLOR }}></span><strong>${rows.length} ${rows.length === 1 ? 'spectrum' : 'spectra'}</strong>
        <span class="num">${flagged.length} may overlap or leave the detector · ${clean.length} clean</span></figcaption>
      <${Legend} items=${mean.length ? [{ name: S.spec.showOverlaps ? 'Each particle' : 'Each clean particle', color: SPECTRUM_COLOR, line: true }, { name: 'Mean of the clean ones', color: THEME.ink, line: true }] : []} />
      ${series.length ? html`<${Chart} height=${190} spec=${{ aria: 'Emission spectra of the fitted particles', xLabel: 'Wavelength (nm)', yLabel: 'pps', xName: 'λ', xUnit: 'nm', yUnit: 'pps', yZero: true, series }} />`
        : html`<p class="hint">Every spectrum here may overlap another. Turn on “Show possible overlaps” to see them, or tap a particle for its own.</p>`}
      <${Toggle} id="spec-overlaps" label="Show possible overlaps" hint="Spectra that another particle’s spectrum may run into, dim neighbours included" checked=${S.spec.showOverlaps}
        onChange=${(v) => { S.spec.showOverlaps = v; bump(); }} />
    </figure>
    <p class="hint">Each spectrum is the 5-row average along the particle’s streak, minus its minimum, as PsyFit’s Get Spectra reads it${file.isMovie ? ', averaged over 5 frames around the one shown' : ''}. The CSV opens in Process Spectra.</p>` : null}
  <//>`;
}

function setHist(patch) {
  Object.assign(S.hist, patch);
  persist();
  bump();
}

function histSpec(set, range) {
  const all = set.values.filter((v) => Number.isFinite(v));
  const vals = range ? all.filter((v) => v >= range[0] && v <= range[1]) : all;
  if (!vals.length) return null;
  const sm = summarize(vals);
  const nb = S.hist.bins > 0 ? S.hist.bins : autoBinCount(vals);
  const hst = histogram(vals, nb);
  const spec = {
    title: `${set.label} brightness`, aria: `${set.label}: histogram of ${vals.length} particle brightnesses`,
    xLabel: 'Brightness (pps)', yLabel: 'Count', bars: { edges: hst.edges, counts: hst.counts, color: set.color },
    series: [], marks: [], bands: [], countLabel: vals.length === 1 ? 'particle' : 'particles',
  };
  if (sm && sm.gmm) {
    const pts = [];
    const lo = hst.edges[0], hi = hst.edges[hst.edges.length - 1];
    for (let i = 0; i <= 160; i++) {
      const x = lo + (hi - lo) * i / 160;
      pts.push({ x, y: gmmPdf(sm.gmm, x) * vals.length * hst.width });
    }
    spec.series.push({ name: 'Mixture fit', color: THEME.ink, points: pts, width: 1.75, hit: false });
    spec.marks.push({ x: sm.mu, label: `μ ${fmtBrightness(sm.mu)}`, color: THEME.accent });
  }
  return { spec, sm, n: vals.length };
}

function ClassBar({ values, single }) {
  const counts = classify(values, single);
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return html`<div class="classes">
    <div class="class-bar" role="img" aria-label=${CLASS_NAMES.map((n, i) => `${n} ${counts[i]}`).join(', ')}>
      ${counts.map((c, i) => (c ? html`<span key=${i} style=${{ flex: `${c} 0 0`, background: CLASS_COLORS[i] }}></span>` : null))}
    </div>
    <div class="class-legend">
      ${CLASS_NAMES.map((n, i) => html`<span key=${n}><span class="dot" style=${{ background: CLASS_COLORS[i] }}></span>${n} ${counts[i]} <small>(${(100 * counts[i] / total).toFixed(0)}%)</small></span>`)}
    </div>
  </div>`;
}

function Histograms() {
  const sets = brightnessSets();
  const [range, setRange] = useState(null);
  if (!sets.length) return html`<p class="hint">No fits yet. Run “Detect and fit” first.</p>`;
  let lo = Infinity, hi = -Infinity;
  for (const s of sets) for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const r = range || [lo, hi];
  return html`<div class="plots">
    <div class="field-row">
      <${Stepper} id="gmm-k" label="Mixture components" value=${S.hist.components} min=${1} max=${4} onChange=${(v) => setHist({ components: v })} />
      <${Seg} label="Summary peak" value=${S.hist.pick} onChange=${(v) => setHist({ pick: v })}
        options=${[{ value: 'weight', label: 'Most populated' }, { value: 'lowest', label: 'Dimmest' }]} />
    </div>
    <div class="field-row">
      <${Num} id="hist-min" label="From" unit="pps" value=${r[0]} onChange=${(v) => setRange([v, r[1]])} />
      <${Num} id="hist-max" label="To" unit="pps" value=${r[1]} onChange=${(v) => setRange([r[0], v])} />
      ${S.files.length > 1 ? html`<${Seg} label="Which files" value=${S.hist.pool} onChange=${(v) => { setHist({ pool: v }); if (v === 'all') runDetection({ allFiles: true }); }}
        options=${[{ value: 'file', label: 'This file' }, { value: 'all', label: 'All open' }]} />` : null}
    </div>
    ${sets.map((set) => {
      const res = r[1] > r[0] ? histSpec(set, r) : null;
      if (!res) return html`<p class="hint" key=${set.key}>${set.label}: no particles in this brightness range.</p>`;
      const { spec, sm, n } = res;
      const cv = sm && Number.isFinite(sm.sigma) && sm.mu ? (100 * sm.sigma / sm.mu) : NaN;
      return html`<figure class="plot" key=${set.key}>
        <figcaption><span class="dot" style=${{ background: set.color }}></span><strong>${set.label}</strong>
          <span class="num">μ ${fmtBrightness(sm.mu)}${Number.isFinite(sm.sigma) ? ` ± ${fmtBrightness(sm.sigma)}` : ''}${Number.isFinite(cv) ? ` · σ/μ ${cv.toFixed(0)}%` : ''} · n ${n}</span></figcaption>
        <${Legend} items=${spec.series.length ? [{ name: `${set.label} particles`, color: set.color }, { name: 'Gaussian mixture fit', color: THEME.ink, line: true }] : []} />
        <${Chart} spec=${spec} height=${170} />
        ${sm && sm.mu > 0 ? html`<${ClassBar} values=${set.values.filter((v) => v >= r[0] && v <= r[1])} single=${summarize(set.values.filter((v) => v >= r[0] && v <= r[1]), S.hist.components, 'lowest').mu} />` : null}
      </figure>`;
    })}
    <p class="hint">μ and σ are one component of a Gaussian mixture fitted to the brightnesses. Classes split at 2×, 3× and 4× the dimmest well-populated peak, as in PsyFit's Monomer Estimation.</p>
  </div>`;
}

/** Brightness in channel X against channel Y for particles found in both. */
function CrossChannel() {
  const file = activeFile();
  const [pair, setPair] = useState(null);
  const [radius, setRadius] = useState(3);
  const sets = brightnessSets('file').filter((s) => s.key !== 'border' && s.key !== 'all');
  if (!file || sets.length < 2) return html`<p class="hint">Fit at least two channels of this file (choose All, or overlay several channels) to compare their brightness.</p>`;
  const x = pair && sets.find((s) => s.key === pair[0]) ? pair[0] : sets[0].key;
  const y = pair && sets.find((s) => s.key === pair[1]) && pair[1] !== x ? pair[1] : sets.find((s) => s.key !== x).key;
  const collect = (ch) => {
    const rect = channelRect(file.scope, ch, file.src);
    const reg = S.display.useReg && S.registration[file.scope] && S.registration[file.scope].channels[ch];
    const list = file.fits.filter((q) => q.ok && q.channel === ch);
    return { list, pts: list.map((q) => (reg ? applyAffine(reg.m, q.x - rect.x, q.y - rect.y) : [q.x - rect.x, q.y - rect.y])) };
  };
  const a = collect(x), b = collect(y);
  const m = matchPoints(a.pts, b.pts, radius);
  const pts = m.pairs.map((p) => ({ x: a.list[p.i].brightness, y: b.list[p.j].brightness }));
  const lr = pts.length >= 2 ? linregress(pts.map((p) => p.x), pts.map((p) => p.y)) : null;
  const xi = channelInfo(file.scope, x), yi = channelInfo(file.scope, y);
  const series = [{ name: 'Particles', color: '#9085e9', points: pts, line: false, markers: true }];
  if (lr) {
    const xs = pts.map((p) => p.x), x0 = Math.min(...xs), x1 = Math.max(...xs);
    series.push({ name: 'Linear fit', color: THEME.ink, points: [{ x: x0, y: lr.intercept + lr.slope * x0 }, { x: x1, y: lr.intercept + lr.slope * x1 }], width: 1.75, dash: [5, 4], hit: false });
  }
  const opts = sets.map((s) => ({ value: s.key, label: s.label }));
  return html`<div class="plots">
    <div class="field-row">
      <label class="field"><span class="field-label">X channel</span>
        <select value=${x} onChange=${(e) => setPair([e.currentTarget.value, y])}>${opts.map((o) => html`<option value=${o.value} key=${o.value}>${o.label}</option>`)}</select></label>
      <label class="field"><span class="field-label">Y channel</span>
        <select value=${y} onChange=${(e) => setPair([x, e.currentTarget.value])}>${opts.filter((o) => o.value !== x).map((o) => html`<option value=${o.value} key=${o.value}>${o.label}</option>`)}</select></label>
      <${Stepper} id="xc-radius" label="Match within (px)" value=${radius} min=${1} max=${10} onChange=${setRadius} />
    </div>
    <figure class="plot">
      <figcaption><strong>${yi.label} against ${xi.label}</strong>
        <span class="num">${pts.length} matched of ${a.list.length} and ${b.list.length}${lr ? ` · slope ${fmtNum(lr.slope, 3)} · R² ${lr.r2.toFixed(3)}` : ''}</span></figcaption>
      <${Legend} items=${lr ? [{ name: 'Matched particles', color: '#9085e9' }, { name: `y = ${fmtNum(lr.slope, 3)}x + ${fmtNum(lr.intercept, 3)}`, color: THEME.ink, line: true }] : []} />
      ${pts.length ? html`<${Chart} height=${220} spec=${{ aria: `${yi.label} brightness against ${xi.label} brightness`, xLabel: `${xi.label} brightness (pps)`, yLabel: `${yi.label} (pps)`, xName: xi.label, yName: yi.label, xUnit: 'pps', yUnit: 'pps', series, yZero: true, xZero: true }} />`
        : html`<p class="hint">No particle of ${xi.label} has a partner in ${yi.label} within ${radius} px. Register the channels in the Overlay panel, or widen the match distance.</p>`}
    </figure>
    <p class="hint">Positions are compared within each channel's own frame${S.display.useReg && S.registration[file.scope] ? ', after applying the stored channel registration' : ' (no registration stored for this microscope)'}.</p>
  </div>`;
}

function FitTable() {
  const file = activeFile();
  if (!file) return null;
  const scene = buildScene();
  const rows = [];
  if (scene) for (const L of scene.layers) for (const q of fitsForChannel(L.file, L.layer.channel)) if (!rows.includes(q)) rows.push(q);
  for (const q of file.manual) rows.push(q);
  if (!rows.length) return html`<p class="hint">No fits yet.</p>`;
  const sel = new Set(S.sel && S.sel.ids ? S.sel.ids : []);
  const shown = rows.slice().sort((a, b) => b.brightness - a.brightness).slice(0, 300);
  return html`<div class="table-wrap">
    <table class="table">
      <thead><tr><th>Ch</th><th>x</th><th>y</th><th>Brightness</th><th>σx nm</th><th>σy nm</th><th>SNR</th></tr></thead>
      <tbody>${shown.map((q) => html`<tr key=${q.id} class=${sel.has(q.id) ? 'is-on' : ''} onClick=${() => select({ ids: [q.id] })}>
        <td><span class="dot" style=${{ background: channelInfo(file.scope, q.channel).color }}></span>${channelInfo(file.scope, q.channel).short}${q.manual ? '*' : ''}</td>
        <td>${q.lx.toFixed(1)}</td><td>${q.ly.toFixed(1)}</td><td>${fmtBrightness(q.brightness)}</td>
        <td>${(q.sx * file.pixUm * 1000).toFixed(0)}</td><td>${(q.sy * file.pixUm * 1000).toFixed(0)}</td><td>${Number.isFinite(q.snr) ? q.snr.toFixed(0) : '–'}</td></tr>`)}</tbody>
    </table>
    ${rows.length > shown.length ? html`<p class="hint">Showing the brightest ${shown.length} of ${rows.length}. The CSV has all of them.</p>` : null}
    <p class="hint">Positions are in the channel's own pixels, as PsyFit reports them. * marks a spot you double-tapped.</p>
  </div>`;
}

export function FitSheet() {
  const file = activeFile();
  const [view, setView] = useState('hist');
  if (!file) return html`<div class="sheet-body"><p class="hint">Open a file to fit its particles.</p></div>`;
  const p = S.fit;
  const sets = file.fitsKey === frameKey(file) ? brightnessSets('file') : [];
  const total = sets.reduce((a, s) => a + s.values.length, 0);
  const exportCsv = async () => {
    const text = fitsCsv(S.hist.pool === 'all' ? 'all' : 'file');
    if (text.split('\n').length < 3) { toast('There are no fits to export yet.', 'warn'); return; }
    try {
      const r = await saveFile(`${file.name.replace(/\.sif$/i, '')}_fits.csv`, text, 'text/csv');
      if (r !== 'declined') toast('Fit table saved as CSV.');
    } catch (err) { toast(err.message, 'error'); }
  };
  return html`<div class="sheet-body">
    <${Section} title="Detect and fit" aside=${html`<span class="count" role="status">${total} ${total === 1 ? 'PSF' : 'PSFs'}</span>`}>
      <${Slider} id="fit-threshold" label="Threshold" min=${0.5} max=${30} step=${0.5} value=${p.threshold}
        onChange=${(v) => setFitParam({ threshold: v })} format=${(v) => `${v} σ`}
        hint="Peaks must rise this many standard deviations above the mean of the smoothed image." />
      <div class="field-row">
        <${Stepper} id="fit-mindist" label="Min spacing (px)" value=${p.minDistance} min=${1} max=${15} onChange=${(v) => setFitParam({ minDistance: v })} />
        <${Stepper} id="fit-sigmax" label="Max σ (µm)" value=${p.sigMaxUm} min=${0.1} max=${0.8} step=${0.05} format=${(v) => v.toFixed(2)} onChange=${(v) => setFitParam({ sigMaxUm: v })} />
      </div>
      <div class="field-row">
        <${Seg} label="Signal type" value=${p.signal} onChange=${(v) => setFitParam({ signal: v })}
          options=${[{ value: 'UCNP', label: 'UCNP (peaks)' }, { value: 'dye', label: 'Dye (blobs)' }]} />
      </div>
      <${Toggle} id="fit-crowded" label="Crowded field" hint="Fit overlapping PSFs together and look for hidden neighbours" checked=${p.crowded} onChange=${(v) => setFitParam({ crowded: v })} />
      <${Toggle} id="fit-live" label="Refit as settings change" checked=${p.live} onChange=${(v) => setFitParam({ live: v })} />
      <div class="btn-row">
        <${Btn} kind="btn-primary" icon="fit" id="run-fit" onClick=${() => { const r = runDetection(); toast(`${r.total} of ${r.candidates} detected peaks passed the fit.`); }}>Detect and fit<//>
        ${S.files.length > 1 ? html`<${Btn} id="run-fit-all" onClick=${() => { const r = runDetection({ allFiles: true }); setHist({ pool: 'all' }); toast(`${r.total} PSFs across ${S.files.length} files.`); }}>Fit all open files<//>` : null}
        <${Btn} kind="btn-quiet" icon="trash" onClick=${() => clearFits()}>Clear<//>
      </div>
      ${sets.length ? html`<ul class="summary">${sets.map((s) => {
        const sm = summarize(s.values);
        return html`<li key=${s.key}><span class="dot" style=${{ background: s.color }}></span><span class="summary-name">${s.label}</span>
          <span class="num">${s.values.length}</span><span class="num">${sm ? fmtBrightness(sm.mu) : '–'}${sm && Number.isFinite(sm.sigma) ? html` <small>± ${fmtBrightness(sm.sigma)}</small>` : ''}</span></li>`;
      })}</ul>` : null}
    <//>
    ${file.scope === 'nikon' ? html`<${SpectraSection} file=${file} />` : null}
    <${Section} title="Results" aside=${html`<${Btn} small=${true} icon="save" onClick=${exportCsv} id="export-fits">CSV<//>`}>
      <${Seg} label="Result view" value=${view} onChange=${setView}
        options=${[{ value: 'hist', label: 'Histogram' }, { value: 'cross', label: 'Channel vs channel' }, { value: 'table', label: 'Table' }]} />
      ${view === 'hist' ? html`<${Histograms} />` : view === 'cross' ? html`<${CrossChannel} />` : html`<${FitTable} />`}
    <//>
  </div>`;
}
