// App shell: top bar, channel chips, image viewer, selection card, colour-scale
// card, movie bar, the panel area and the tab bar.

import { h, render } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { html, Btn, IconBtn, Icon, Facts, Seg } from './widgets.js';
import { Viewer } from './viewer.js';
import { FilesSheet } from './sheet_files.js';
import { DisplaySheet, Contrast } from './sheet_display.js';
import { FitSheet } from './sheet_fit.js';
import { OverlaySheet } from './sheet_overlay.js';
import { MovieSheet } from './sheet_movie.js';
import { fmtBrightness, fmtNum } from './theme.js';
import {
  S, subscribe, bump, toast, persist, activeFile, fileById, setSheet, selectChannel, setOverlay, setScope, setFrame, setView,
  togglePlayback, stopPlayback, addPickedFiles, loadDemo, findFit, removeFit, select, locPrecisionNm, runTrace, frameKey,
  setScaleEditor,
} from '../store.js';
import { SCOPES, SCOPE_ORDER, availableChannels, channelInfo } from '../core/layouts.js';
import { hueCss } from '../core/render.js';
import { saveFile, loadPrefs, savePrefs } from '../platform.js';

let viewer = null;

async function saveImage() {
  const file = activeFile();
  if (!viewer || !file) return;
  try {
    const blob = await viewer.toBlob();
    if (!blob) throw new Error('The image could not be rendered.');
    const r = await saveFile(`${file.name.replace(/\.sif$/i, '')}_${frameKey(file)}.png`, blob, 'image/png');
    if (r !== 'declined') toast('Image saved.');
  } catch (err) { toast(err.message, 'error'); }
}

function TopBar() {
  const file = activeFile();
  const i = file ? file.src.info : null;
  return html`<header class="topbar">
    <button type="button" class="file-switch" onClick=${() => setSheet('files')} aria-label=${file ? `${file.name}. Open the Files panel` : 'Open a file'}>
      <span class="file-name">${file ? file.name : 'PsyFit Mobile'}</span>
      <span class="file-meta">${file
        ? `${file.origin.kind === 'demo' ? 'Simulated demo · ' : ''}${file.src.width}×${file.src.height}${file.isMovie ? ` · ${file.src.frames} frames` : ''} · ${fmtNum(i.ExposureTime * 1000, 3)} ms${i.AccumulatedCycles > 1 ? `×${i.AccumulatedCycles}` : ''} · gain ${i.GainDAC}`
        : 'Andor .sif viewer and single-molecule analysis'}</span>
    </button>
    ${file ? html`<label class="scope-chip"><span class="visually-hidden">Microscope</span>
      <select value=${file.scope} onChange=${(e) => setScope(file, e.currentTarget.value)} id="scope-select">
        ${SCOPE_ORDER.map((id) => html`<option value=${id} key=${id}>${SCOPES[id].name}</option>`)}
      </select><${Icon} name="chevron" size=${14} /></label>` : null}
    <${IconBtn} icon="expand" title="Fit the image to the screen" onClick=${() => viewer && viewer.fitView()} id="fit-view" />
    <${IconBtn} icon="plus" title="Open files" onClick=${() => { const el = document.getElementById('quick-file-input'); if (el) el.click(); }} id="quick-open" />
    <input class="visually-hidden" id="quick-file-input" type="file" multiple tabindex="-1" aria-hidden="true"
      onChange=${async (e) => { const files = e.currentTarget.files; await addPickedFiles(files); e.currentTarget.value = ''; }} />
  </header>`;
}

function ChannelBar() {
  const file = activeFile();
  if (!file) return null;
  const chans = availableChannels(file.scope, file.src);
  const mine = S.layers.filter((l) => l.fileId === file.id);
  const on = new Set(mine.map((l) => l.channel));
  const hueOf = (id) => { const l = mine.find((x) => x.channel === id); return l ? hueCss(l.hue) : null; };
  return html`<div class="chips" role="toolbar" aria-label="Channels">
    <div class="chip-scroll">
      <button type="button" class=${`chip${on.has('all') ? ' is-on' : ''}`} aria-pressed=${String(on.has('all'))} onClick=${() => selectChannel('all')} id="chip-all">All</button>
      ${chans.map((c) => html`<button type="button" key=${c.id} id=${`chip-${c.id}`} class=${`chip${on.has(c.id) ? ' is-on' : ''}`} aria-pressed=${String(on.has(c.id))}
        onClick=${() => selectChannel(c.id)}><span class="dot" style=${{ background: S.overlay && on.has(c.id) ? hueOf(c.id) : c.color }}></span>${c.label}</button>`)}
    </div>
    ${chans.length > 1 || S.files.length > 1 ? html`<button type="button" class=${`chip chip-overlay${S.overlay ? ' is-on' : ''}`} aria-pressed=${String(S.overlay)}
      title="Overlay several channels" aria-label="Overlay several channels" onClick=${() => setOverlay(!S.overlay)} id="chip-overlay"><${Icon} name="layers" size=${17} /><span class="chip-overlay-text">Overlay</span></button>` : null}
  </div>`;
}

function FitCard({ fits }) {
  const first = fits[0];
  const file = fileById(first.fileId);
  if (!file) return null;
  const p = file.pixUm;
  const close = () => select(null);
  const many = fits.length > 1;
  const flags = (q) => [!q.sigmaOk ? `wider than ${S.fit.sigMaxUm} µm` : null, !q.brightnessOk ? 'brightness out of range' : null].filter(Boolean);
  return html`<aside class="card" aria-label="Fit details" id="fit-card">
    <div class="card-head">
      <span class="card-title">${many ? `PSF in ${fits.length} layers` : `PSF · ${channelInfo(file.scope, first.channel).label}`}${first.manual ? '' : ' · auto'}</span>
      ${file.isMovie ? html`<${Btn} small=${true} icon="chart" id="card-trace" onClick=${() => { runTrace(first, S.trace ? S.trace.penalty : 1); if (S.sheet !== 'movie') setSheet('movie'); }}>Trace<//>` : null}
      <${IconBtn} icon="trash" title="Remove this fit" onClick=${() => fits.forEach((q) => removeFit(q.id))} size=${18} />
      <${IconBtn} icon="close" title="Close" onClick=${close} size=${18} />
    </div>
    ${many ? html`<table class="table card-table">
      <thead><tr><th>Layer</th><th>Brightness</th><th>σx nm</th><th>σy nm</th><th>SNR</th></tr></thead>
      <tbody>${fits.map((q) => {
        const f = fileById(q.fileId);
        return html`<tr key=${q.id} class=${q.ok ? '' : 'is-dim'}><td><span class="dot" style=${{ background: channelInfo(f.scope, q.channel).color }}></span>${channelInfo(f.scope, q.channel).label}</td>
          <td>${fmtBrightness(q.brightness)}</td><td>${(q.sx * f.pixUm * 1000).toFixed(0)}</td><td>${(q.sy * f.pixUm * 1000).toFixed(0)}</td><td>${Number.isFinite(q.snr) ? q.snr.toFixed(0) : '–'}</td></tr>`;
      })}</tbody></table>`
      : html`<div class="card-main">
        <div class="hero"><span class="hero-num" id="fit-brightness">${fmtBrightness(first.brightness)}</span><span class="hero-label">brightness, background removed</span></div>
        <${Facts} rows=${[
          ['σx / σy', `${(first.sx * p * 1000).toFixed(0)} / ${(first.sy * p * 1000).toFixed(0)} nm (${first.sx.toFixed(2)} / ${first.sy.toFixed(2)} px)`],
          ['FWHM', `${(2.3548 * Math.sqrt(first.sx * first.sy) * p * 1000).toFixed(0)} nm`],
          ['Fit volume', fmtBrightness(first.brightnessFit)],
          ['Peak / background', `${fmtBrightness(first.amp)} / ${fmtBrightness(first.offset)} per px`],
          ['SNR · R²', `${Number.isFinite(first.snr) ? first.snr.toFixed(1) : '–'} · ${Number.isFinite(first.r2) ? first.r2.toFixed(3) : '–'}`],
          ['Position', `${first.lx.toFixed(2)}, ${first.ly.toFixed(2)} px${first.run === 'all' ? '' : ` (sensor ${first.x.toFixed(1)}, ${first.y.toFixed(1)})`}`],
          ['Localization', `± ${fmtNum(locPrecisionNm(first, file), 2)} nm (photon-limited estimate)`],
          first.neighbors ? ['Co-fitted neighbours', String(first.neighbors)] : null,
        ]} />
        ${flags(first).length ? html`<p class="warn">Rejected by the fit filters: ${flags(first).join(', ')}.</p>` : null}
      </div>`}
  </aside>`;
}

function TrackCard({ file, track }) {
  const st = track.stats;
  return html`<aside class="card" aria-label="Track details" id="track-card">
    <div class="card-head">
      <span class="card-title">Track ${track.id} · ${channelInfo(file.scope, track.channel).label}</span>
      <${Btn} small=${true} icon="chart" onClick=${() => { if (S.sheet !== 'movie') setSheet('movie'); }}>MSD<//>
      <${IconBtn} icon="close" title="Close" onClick=${() => select(null)} size=${18} />
    </div>
    <div class="card-main">
      <div class="hero"><span class="hero-num">${fmtNum(st.D, 3)}</span><span class="hero-label">µm²/s diffusion coefficient</span></div>
      <${Facts} rows=${[
        ['Length', `${st.points} frames, ${fmtNum(st.durationS, 3)} s`],
        ['Net / path', `${st.netUm.toFixed(2)} / ${st.pathUm.toFixed(2)} µm`],
        ['Anomalous exponent α', Number.isFinite(st.alpha) ? st.alpha.toFixed(2) : '–'],
        ['Mean brightness', fmtBrightness(st.meanBrightness)],
      ]} />
    </div>
  </aside>`;
}

/** Opens from a tap on the colour bar: min and max of every layer shown. */
function ScaleCard() {
  const layers = S.layers.filter((l) => l.visible && fileById(l.fileId));
  if (!layers.length) return null;
  const multi = layers.length > 1;
  const setLog = (v) => { S.display.log = v === 'log'; persist(); bump('scene'); };
  return html`<aside class="card card-scale" aria-label="Colour scale" id="scale-card">
    <div class="card-head">
      <span class="card-title">Colour scale</span>
      <${Seg} label="Intensity scale" value=${S.display.log ? 'log' : 'lin'} onChange=${setLog}
        options=${[{ value: 'lin', label: 'Linear' }, { value: 'log', label: 'Log' }]} />
      <${IconBtn} icon="close" title="Close" onClick=${() => setScaleEditor(false)} size=${18} id="scale-close" />
    </div>
    <div class="card-scale-body">
      ${layers.map((l) => html`<${Contrast} key=${l.id} layer=${l} multi=${multi} idPrefix="sc-" compact=${true} />`)}
    </div>
  </aside>`;
}

function SelectionCard() {
  if (S.scale) return html`<${ScaleCard} />`;
  if (!S.sel) return null;
  if (S.sel.ids) {
    const fits = S.sel.ids.map(findFit).filter(Boolean);
    return fits.length ? html`<${FitCard} fits=${fits} />` : null;
  }
  if (S.sel.trackId) {
    const file = fileById(S.sel.fileId);
    const track = file && file.tracks ? file.tracks.list.find((t) => t.id === S.sel.trackId) : null;
    return track ? html`<${TrackCard} file=${file} track=${track} />` : null;
  }
  return null;
}

function MovieBar() {
  const file = activeFile();
  if (!file || file.loaded.length < 2) return null;
  const n = file.loaded.length;
  const label = file.view === 'frame' ? `${file.loaded[file.cur] + 1} / ${file.src.frames}` : (file.view === 'mean' ? 'Average' : 'Maximum');
  return html`<div class="moviebar" role="group" aria-label="Movie playback">
    <${IconBtn} icon="prev" title="Previous frame" onClick=${() => { stopPlayback(); setFrame(file, file.cur - 1); }} disabled=${file.view === 'frame' && file.cur <= 0} />
    <${IconBtn} icon=${S.playing ? 'pause' : 'play'} title=${S.playing ? 'Pause' : 'Play'} onClick=${() => togglePlayback()} id="play" />
    <${IconBtn} icon="next" title="Next frame" onClick=${() => { stopPlayback(); setFrame(file, file.cur + 1); }} disabled=${file.view === 'frame' && file.cur >= n - 1} />
    <label class="visually-hidden" for="frame-slider">Frame</label>
    <input id="frame-slider" type="range" min="0" max=${n - 1} step="1" value=${file.cur}
      onInput=${(e) => { stopPlayback(); if (file.view !== 'frame') setView(file, 'frame'); setFrame(file, parseInt(e.currentTarget.value, 10)); }} />
    <output class="frame-label" for="frame-slider">${label}</output>
  </div>`;
}

const TABS = [
  ['files', 'Files', 'files'], ['display', 'Display', 'display'], ['analyze', 'Fit', 'fit'],
  ['overlay', 'Overlay', 'layers'], ['movie', 'Movie', 'movie'],
];
const TITLES = { files: 'Files', display: 'Display', analyze: 'Fit', overlay: 'Overlay', movie: 'Movie' };

function SheetHost() {
  if (!S.sheet) return null;
  const body = S.sheet === 'files' ? html`<${FilesSheet} />`
    : S.sheet === 'display' ? html`<${DisplaySheet} onSaveImage=${saveImage} />`
      : S.sheet === 'analyze' ? html`<${FitSheet} />`
        : S.sheet === 'overlay' ? html`<${OverlaySheet} />` : html`<${MovieSheet} />`;
  return html`<section class="sheet" aria-label=${`${TITLES[S.sheet]} panel`} id=${`sheet-${S.sheet}`}>
    <div class="sheet-head"><h2>${TITLES[S.sheet]}</h2><${IconBtn} icon="close" title="Close panel" onClick=${() => setSheet(S.sheet)} id="sheet-close" /></div>
    ${body}
  </section>`;
}

function TabBar() {
  return html`<nav class="tabbar" aria-label="Panels">
    ${TABS.map(([id, label, icon]) => html`<button type="button" key=${id} id=${`tab-${id}`} class=${`tab${S.sheet === id ? ' is-on' : ''}`}
      aria-pressed=${String(S.sheet === id)} onClick=${() => setSheet(id)}><${Icon} name=${icon} size=${21} /><span>${label}</span></button>`)}
  </nav>`;
}

function Status() {
  const b = S.busy;
  return html`<div class="status" aria-live="polite">
    ${b ? html`<div class="busy" role="status">
      <span class="busy-label">${b.label}${b.total > 1 ? ` · ${b.done} of ${b.total}` : ''}</span>
      <progress max=${b.total || 1} value=${b.total > 1 ? b.done : undefined}></progress>
      ${b.cancel ? html`<${Btn} small=${true} onClick=${b.cancel}>Cancel<//>` : null}
    </div>` : null}
    ${S.toast ? html`<div class=${`toast is-${S.toast.kind}`} role=${S.toast.kind === 'error' ? 'alert' : 'status'} onClick=${() => { S.toast = null; bump(); }}>${S.toast.text}</div>` : null}
  </div>`;
}

function Hint() {
  const [seen, setSeen] = useState(() => !!loadPrefs().hintSeen);
  const file = activeFile();
  const probed = S.files.some((f) => f.manual.length);
  useEffect(() => {
    if (probed && !seen) { setSeen(true); savePrefs({ ...loadPrefs(), hintSeen: true }); }
  }, [probed]);
  if (!file || seen || S.sel) return null;
  return html`<div class="hintchip" role="note">Double-tap a spot to fit it · hold for pixel values</div>`;
}

function Welcome() {
  return html`<div class="welcome">
    <h1>PsyFit Mobile</h1>
    <p>Open an Andor <code>.sif</code> image or movie to fit PSF brightness, compare channels and follow particles.</p>
    <div class="btn-row">
      <${Btn} kind="btn-primary" icon="files" onClick=${() => setSheet('files')}>Open a file<//>
      <${Btn} onClick=${() => loadDemo('image')}>Load the demo<//>
    </div>
  </div>`;
}

function App() {
  const [, force] = useState(0);
  const hostRef = useRef(null);
  useEffect(() => subscribe(() => force((n) => (n + 1) % 1e9)), []);
  useEffect(() => {
    viewer = new Viewer(hostRef.current);
    if (typeof window !== 'undefined') window.__psyfit = { S, viewer };
    return () => { viewer.destroy(); viewer = null; };
  }, []);
  useEffect(() => {
    const key = (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      const file = activeFile();
      if (!file) return;
      if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
      else if (e.key === 'ArrowRight') { stopPlayback(); setFrame(file, file.cur + 1); }
      else if (e.key === 'ArrowLeft') { stopPlayback(); setFrame(file, file.cur - 1); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, []);
  const drop = (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) addPickedFiles(e.dataTransfer.files);
  };
  const file = activeFile();
  return html`<div class=${`app${S.sheet ? ' has-sheet' : ''}`} onDragOver=${(e) => e.preventDefault()} onDrop=${drop}>
    <${TopBar} />
    <${ChannelBar} />
    <div class="workspace">
      <main class="stage">
        <div class="viewer" ref=${hostRef}></div>
        ${file ? null : html`<${Welcome} />`}
        <${Hint} />
        <${SelectionCard} />
        <${Status} />
      </main>
      <${SheetHost} />
    </div>
    <${MovieBar} />
    <${TabBar} />
  </div>`;
}

export function mount(el) {
  render(html`<${App} />`, el);
}

export { persist, h };
