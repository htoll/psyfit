// Movie panel: which frames are in memory, playback, particle tracking and
// per-spot intensity traces with step detection.

import { useState } from 'preact/hooks';
import { html, Btn, Seg, Slider, Stepper, Toggle, Section, Num, Facts } from './widgets.js';
import { Chart, Legend } from './plots.js';
import { THEME, fmtBrightness, fmtNum, fmtBytes, fmtDuration } from './theme.js';
import {
  S, bump, persist, toast, activeFile, loadWindow, maxFramesFor, setView, frameDt, runTracking, clearTracks, tracksCsv,
  select, findFit, runTrace, updateTraceSteps, traceCsv, fileById,
} from '../store.js';
import { channelInfo } from '../core/layouts.js';
import { trackColor } from '../core/tracking.js';
import { median } from '../core/stats.js';
import { saveFile } from '../platform.js';

function setMovie(patch, kind = 'overlay') {
  Object.assign(S.movie, patch);
  persist();
  bump(kind);
}

function FrameLoader({ file }) {
  const total = file.src.frames;
  const cap = maxFramesFor(file);
  const win = file.window || { start: 0, step: 1, count: file.loaded.length };
  const [draft, setDraft] = useState(null);
  const d = draft || { start: win.start + 1, count: win.count, step: win.step };
  const room = Math.floor((total - d.start) / d.step) + 1;
  const count = Math.max(1, Math.min(d.count, room, cap));
  const last = d.start + (count - 1) * d.step;
  const bytes = count * file.src.width * file.src.height * 4;
  const apply = async (w) => {
    try { await loadWindow(file, { start: w.start - 1, step: w.step, count: w.count }); setDraft(null); }
    catch (err) { toast(err.message, 'error'); }
  };
  return html`<div>
    <p class="hint">In memory: frames ${win.start + 1} to ${win.start + 1 + (win.count - 1) * win.step}${win.step > 1 ? `, every ${win.step}` : ''} of ${total} (${fmtBytes(win.count * file.src.width * file.src.height * 4)}). Only these are read from the file.</p>
    <div class="field-row">
      <${Num} id="win-start" label="First frame" value=${d.start} min=${1} max=${total} digits=${7} onChange=${(v) => setDraft({ ...d, start: Math.round(v) })} />
      <${Num} id="win-count" label="How many" value=${d.count} min=${1} max=${cap} digits=${7} onChange=${(v) => setDraft({ ...d, count: Math.round(v) })} />
      <${Num} id="win-step" label="Every" value=${d.step} min=${1} max=${Math.max(1, total - 1)} digits=${7} onChange=${(v) => setDraft({ ...d, step: Math.round(v) })} />
    </div>
    <div class="btn-row">
      <${Btn} kind="btn-primary" id="load-frames" onClick=${() => apply({ start: d.start, step: d.step, count })}>Load ${count} frames (${fmtBytes(bytes)})<//>
      <${Btn} small=${true} onClick=${() => setDraft({ start: 1, step: 1, count: Math.min(total, cap) })}>First ${Math.min(total, cap)}<//>
      ${total > cap ? html`<${Btn} small=${true} onClick=${() => setDraft({ start: 1, step: Math.ceil(total / cap), count: cap })}>Spread over all<//>` : null}
    </div>
    <p class="hint">Frames ${d.start} to ${last}. Up to ${cap} frames of this size fit in the ${S.movie.budgetMB} MB limit.</p>
    <${Seg} label="Memory limit" value=${S.movie.budgetMB} onChange=${(v) => setMovie({ budgetMB: v }, 'ui')}
      options=${[{ value: 80, label: '80 MB' }, { value: 160, label: '160 MB' }, { value: 320, label: '320 MB' }, { value: 640, label: '640 MB' }]} />
    <p class="hint">Raise the limit only on a phone with memory to spare: Safari closes pages that use too much.</p>
  </div>`;
}

function TrackPanel({ file }) {
  const tr = file.tracks;
  const exportTracks = async () => {
    try { const r = await saveFile(`${file.name.replace(/\.sif$/i, '')}_tracks.csv`, tracksCsv(file), 'text/csv'); if (r !== 'declined') toast('Tracks saved as CSV.'); }
    catch (err) { toast(err.message, 'error'); }
  };
  const sel = S.sel && S.sel.trackId && S.sel.fileId === file.id ? tr && tr.list.find((t) => t.id === S.sel.trackId) : null;
  const ds = tr ? tr.list.map((t) => t.stats.D).filter((v) => Number.isFinite(v)) : [];
  return html`<div>
    <div class="field-row">
      <${Stepper} id="trk-disp" label="Max step (px/frame)" value=${S.track.maxDisp} min=${1} max=${30} onChange=${(v) => { S.track.maxDisp = v; persist(); bump(); }} />
      <${Stepper} id="trk-gap" label="Bridge gaps (frames)" value=${S.track.maxGap} min=${0} max=${10} onChange=${(v) => { S.track.maxGap = v; persist(); bump(); }} />
      <${Stepper} id="trk-min" label="Min length (frames)" value=${S.track.minLength} min=${2} max=${100} onChange=${(v) => { S.track.minLength = v; persist(); bump(); }} />
    </div>
    <div class="btn-row">
      <${Btn} kind="btn-primary" icon="track" id="run-tracking" onClick=${() => runTracking()}>Track particles<//>
      ${tr ? html`<${Btn} small=${true} icon="save" onClick=${exportTracks}>CSV<//>` : null}
      ${tr ? html`<${Btn} small=${true} kind="btn-quiet" icon="trash" onClick=${() => clearTracks(file)}>Clear<//>` : null}
    </div>
    <p class="hint">Uses the Fit panel's threshold and spacing on every loaded frame of the channels shown, then links nearby detections frame to frame.</p>
    <${Slider} id="trk-tail" label="Tail" min=${0} max=${100} step=${1} value=${Math.min(S.movie.tail, 100)}
      format=${(v) => (v === 0 ? 'whole path' : `${v} frames`)} onChange=${(v) => setMovie({ tail: v })} />
    <div class="field-row">
      <${Seg} label="Track colour" value=${S.movie.colorMode} onChange=${(v) => setMovie({ colorMode: v })}
        options=${[{ value: 'random', label: 'One colour per track' }, { value: 'channel', label: 'By channel' }]} />
    </div>
    <${Toggle} id="trk-show" label="Show tracks" checked=${S.movie.showTracks} onChange=${(v) => setMovie({ showTracks: v })} />
    ${tr ? html`<div class="stat-row">
      <div class="stat"><span class="stat-num">${tr.list.length}</span><span class="stat-label">tracks</span></div>
      <div class="stat"><span class="stat-num">${ds.length ? fmtNum(median(ds), 2) : '–'}</span><span class="stat-label">median D, µm²/s</span></div>
      <div class="stat"><span class="stat-num">${tr.list.length ? Math.round(median(tr.list.map((t) => t.points.length))) : '–'}</span><span class="stat-label">median length, frames</span></div>
    </div>` : null}
    ${sel ? html`<${TrackDetail} file=${file} track=${sel} />` : (tr && tr.list.length ? html`<p class="hint">Tap a track on the image, or a row below, for its path statistics.</p>` : null)}
    ${tr && tr.list.length ? html`<div class="table-wrap"><table class="table">
      <thead><tr><th>#</th><th>Ch</th><th>Frames</th><th>D µm²/s</th><th>α</th><th>Net µm</th></tr></thead>
      <tbody>${tr.list.slice().sort((a, b) => b.points.length - a.points.length).slice(0, 80).map((t) => html`<tr key=${t.id}
        class=${sel && sel.id === t.id ? 'is-on' : ''} onClick=${() => select({ trackId: t.id, fileId: file.id })}>
        <td><span class="dot" style=${{ background: S.movie.colorMode === 'channel' ? channelInfo(file.scope, t.channel).color : trackColor(t.id) }}></span>${t.id}</td>
        <td>${channelInfo(file.scope, t.channel).short}</td><td>${t.points.length}</td><td>${fmtNum(t.stats.D, 3)}</td>
        <td>${Number.isFinite(t.stats.alpha) ? t.stats.alpha.toFixed(2) : '–'}</td><td>${t.stats.netUm.toFixed(2)}</td></tr>`)}</tbody>
    </table></div>` : null}
  </div>`;
}

function TrackDetail({ file, track }) {
  const st = track.stats;
  const dt = file.tracks.dt;
  const pts = st.msd.map((m) => ({ x: m.lag * dt, y: m.msd * file.pixUm * file.pixUm }));
  const series = [{ name: 'Measured', color: '#3987e5', points: pts, markers: true, line: false }];
  if (Number.isFinite(st.D) && pts.length > 1) {
    const c = 4 * st.locPrecisionUm * st.locPrecisionUm;
    const x1 = pts[Math.min(pts.length - 1, 5)].x;
    series.push({ name: '4Dt fit', color: THEME.ink, width: 1.75, dash: [5, 4], hit: false, points: [{ x: 0, y: c }, { x: x1, y: c + 4 * st.D * x1 }] });
  }
  return html`<figure class="plot">
    <figcaption><span class="dot" style=${{ background: trackColor(track.id) }}></span><strong>Track ${track.id}</strong>
      <span class="num">D ${fmtNum(st.D, 3)} µm²/s${Number.isFinite(st.alpha) ? ` · α ${st.alpha.toFixed(2)}` : ''}</span></figcaption>
    <${Legend} items=${series.length > 1 ? [{ name: 'Mean-squared displacement', color: '#3987e5' }, { name: '4Dt fit to the first lags', color: THEME.ink, line: true }] : []} />
    <${Chart} height=${170} spec=${{ aria: `Mean-squared displacement of track ${track.id}`, xLabel: 'Lag (s)', yLabel: 'MSD (µm²)', xName: 'lag', yName: 'MSD', xUnit: 's', yUnit: 'µm²', series, yZero: true, xZero: true }} />
    <${Facts} rows=${[
      ['Frames', `${st.points} (${file.loaded[st.firstFrame] + 1} to ${file.loaded[st.lastFrame] + 1})`],
      ['Duration', fmtDuration(st.durationS)],
      ['Path / net', `${st.pathUm.toFixed(2)} / ${st.netUm.toFixed(2)} µm`],
      ['Mean speed', `${fmtNum(st.speedUmS, 3)} µm/s`],
      ['Mean brightness', fmtBrightness(st.meanBrightness)],
      ['Localization (MSD offset)', `${(st.locPrecisionUm * 1000).toFixed(0)} nm`],
    ]} />
  </figure>`;
}

function TracePanel({ file }) {
  const tr = S.trace && S.trace.fileId === file.id ? S.trace : null;
  const selFit = S.sel && S.sel.ids ? S.sel.ids.map(findFit).filter((q) => q && q.fileId === file.id)[0] : null;
  const exportTrace = async () => {
    try { const r = await saveFile(`${file.name.replace(/\.sif$/i, '')}_trace.csv`, traceCsv(), 'text/csv'); if (r !== 'declined') toast('Trace saved as CSV.'); }
    catch (err) { toast(err.message, 'error'); }
  };
  const spec = tr ? {
    aria: 'Brightness of the selected spot in every loaded frame, with detected steps',
    xLabel: 'Time (s)', yLabel: 'Brightness (pps)', xName: 't', xUnit: 's', yUnit: 'pps', yZero: true,
    series: [
      { name: 'Brightness', color: '#3987e5', points: tr.values.map((v, i) => ({ x: tr.t[i], y: v })), width: 1.5 },
      { name: 'Steps', color: THEME.accent, points: tr.levels.map((v, i) => ({ x: tr.t[i], y: v })), step: true, width: 2, hit: false },
    ],
  } : null;
  const drops = tr ? tr.steps.filter((s) => s.drop > 0) : [];
  return html`<div>
    <p class="hint">${selFit ? 'Fits the selected spot at its fixed position in every loaded frame.' : 'Double-tap a spot (or tap a fit circle), then plot its brightness over time to see bleaching steps and blinking.'}</p>
    <div class="btn-row">
      <${Btn} kind="btn-primary" icon="chart" id="run-trace" disabled=${!selFit} onClick=${() => { runTrace(selFit, tr ? tr.penalty : 1); }}>Plot brightness over time<//>
      ${tr ? html`<${Btn} small=${true} icon="save" onClick=${exportTrace}>CSV<//>` : null}
    </div>
    ${tr ? html`<figure class="plot">
      <figcaption><span class="dot" style=${{ background: channelInfo(file.scope, tr.channel).color }}></span><strong>Spot at ${tr.x.toFixed(1)}, ${tr.y.toFixed(1)}</strong>
        <span class="num">${tr.steps.length} ${tr.steps.length === 1 ? 'step' : 'steps'}${Number.isFinite(tr.unit) ? ` · typical drop ${fmtBrightness(tr.unit)}` : ''}</span></figcaption>
      <${Legend} items=${[{ name: 'Brightness per frame', color: '#3987e5', line: true }, { name: 'Detected plateaus', color: THEME.accent, line: true }]} />
      <${Chart} spec=${spec} height=${190} />
      <${Slider} id="trace-penalty" label="Step strictness" min=${0.5} max=${6} step=${0.25} value=${tr.penalty} format=${(v) => `${v}×`}
        onChange=${(v) => updateTraceSteps(v)} hint="Kalafut-Visscher detection. Higher keeps only the clearest steps." />
      ${drops.length ? html`<div class="table-wrap"><table class="table">
        <thead><tr><th>At</th><th>From</th><th>To</th><th>Drop</th></tr></thead>
        <tbody>${tr.steps.map((s, i) => html`<tr key=${i}><td>${fmtDuration(s.index * tr.dt)}</td><td>${fmtBrightness(s.before)}</td><td>${fmtBrightness(s.after)}</td><td>${fmtBrightness(s.drop)}</td></tr>`)}</tbody>
      </table></div>` : null}
    </figure>` : null}
  </div>`;
}

export function MovieSheet() {
  const file = activeFile();
  if (!file) return html`<div class="sheet-body"><p class="hint">Open a movie to play it and track particles.</p></div>`;
  if (!file.isMovie) {
    return html`<div class="sheet-body"><${Section} title="Movie">
      <p class="hint">${file.name} is a single frame. Open a kinetic series (a .sif with many frames) to play it back, track particles and plot bleaching steps. “Demo movie” in the Files panel has one to try.</p>
    <//></div>`;
  }
  const dt = frameDt(file);
  return html`<div class="sheet-body">
    <${Section} title="Frames">
      <${FrameLoader} file=${file} />
    <//>
    <${Section} title="Playback">
      <${Slider} id="fps" label="Speed" min=${1} max=${60} step=${1} value=${S.movie.fps} format=${(v) => `${v} fps (${fmtNum(v * dt, 2)}× real time)`}
        onChange=${(v) => setMovie({ fps: v }, 'ui')} />
      <div class="field-row">
        <${Seg} label="What to show" value=${file.view} onChange=${(v) => setView(file, v)}
          options=${[{ value: 'frame', label: 'Frame' }, { value: 'mean', label: 'Average' }, { value: 'max', label: 'Maximum' }]} />
      </div>
      <${Toggle} id="loop" label="Loop" checked=${S.movie.loop} onChange=${(v) => setMovie({ loop: v }, 'ui')} />
      <p class="hint">Frame interval ${fmtDuration(dt)}. Average and Maximum combine the loaded frames; fits then run on that image.</p>
    <//>
    <${Section} title="Tracking">
      <${TrackPanel} file=${file} />
    <//>
    <${Section} title="Intensity trace">
      <${TracePanel} file=${file} />
    <//>
  </div>`;
}

export { fileById };
