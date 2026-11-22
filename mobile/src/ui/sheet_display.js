// Display panel: microscope preset, colormap, contrast and overlay toggles.

import { html, Btn, Seg, Num, Toggle, Section, ContrastBar } from './widgets.js';
import { fmtNum } from './theme.js';
import {
  S, bump, persist, activeFile, setScope, setPixelSize, updateLayer, layerStats, autoRange, layerLabel,
} from '../store.js';
import { SCOPES, SCOPE_ORDER, channelInfo } from '../core/layouts.js';
import { COLORMAPS, lutCss, hueCss } from '../core/render.js';

function setDisplay(patch, kind = 'scene') {
  Object.assign(S.display, patch);
  persist();
  bump(kind);
}

function Contrast({ layer, multi }) {
  const st = layerStats(layer);
  if (!st) return null;
  const file = activeFile();
  const color = multi ? hueCss(layer.hue) : '#aab3c5';
  const set = (lo, hi) => updateLayer(layer.id, { lo, hi, auto: false });
  return html`<div class="contrast-block">
    ${multi ? html`<div class="contrast-name"><span class="dot" style=${{ background: color }}></span>${layerLabel(layer)}</div>` : null}
    <${ContrastBar} stats=${st} lo=${layer.lo} hi=${layer.hi} color=${color} onChange=${set} />
    <div class="field-row">
      <${Num} id=${`lo-${layer.id}`} label="Black" unit="pps" value=${layer.lo} onChange=${(v) => set(Math.min(v, layer.hi - 1e-6), layer.hi)} />
      <${Num} id=${`hi-${layer.id}`} label="White" unit="pps" value=${layer.hi} onChange=${(v) => set(layer.lo, Math.max(v, layer.lo + 1e-6))} />
      <div class="btn-col">
        <${Btn} small=${true} kind=${layer.auto ? 'btn-on' : ''} onClick=${() => updateLayer(layer.id, { auto: true })}>Auto<//>
        <${Btn} small=${true} onClick=${() => set(st.min, st.max)}>Full<//>
      </div>
    </div>
    ${file && !multi ? html`<p class="hint">Image range ${fmtNum(st.min, 4)} to ${fmtNum(st.max, 4)} pps. Auto shows from the 1st percentile up to where the particles sit.</p>` : null}
  </div>`;
}

export function DisplaySheet({ onSaveImage }) {
  const file = activeFile();
  if (!file) return html`<div class="sheet-body"><p class="hint">Open a file to adjust how it is shown.</p></div>`;
  const layers = S.layers.filter((l) => l.visible);
  const multi = layers.length > 1;
  const d = S.display;
  return html`<div class="sheet-body">
    <${Section} title="Microscope">
      <${Seg} label="Microscope" value=${file.scope} onChange=${(v) => setScope(file, v)}
        options=${SCOPE_ORDER.map((id) => ({ value: id, label: SCOPES[id].name }))} />
      <p class="hint">${SCOPES[file.scope].note}${file.scopeGuessed ? '' : '. Not named in the file, so the last choice is used'}.</p>
      <div class="field-row">
        <${Num} id="pix-size" label="Pixel size" unit="µm" value=${file.pixUm} min=${0.005} max=${10} digits=${4} onChange=${(v) => setPixelSize(file, v)} />
      </div>
    <//>
    ${!multi ? html`<${Section} title="Colormap">
      <div class="swatches" role="group" aria-label="Colormap">
        ${COLORMAPS.map((name) => html`<button type="button" key=${name} class=${`swatch${d.cmap === name ? ' is-on' : ''}`}
          aria-pressed=${String(d.cmap === name)} onClick=${() => setDisplay({ cmap: name })}>
          <span class="swatch-ramp" style=${{ background: lutCss(name) }}></span><span>${name}</span></button>`)}
      </div>
    <//>` : null}
    <${Section} title="Contrast" aside=${html`<${Seg} label="Intensity scale" value=${d.log ? 'log' : 'lin'} onChange=${(v) => setDisplay({ log: v === 'log' })}
        options=${[{ value: 'lin', label: 'Linear' }, { value: 'log', label: 'Log' }]} />`}>
      ${layers.map((l) => html`<${Contrast} key=${l.id} layer=${l} multi=${multi} />`)}
      ${file.isMovie ? html`<${Toggle} id="per-frame-auto" label="Re-scale every frame" hint="Off keeps one scale for the whole movie"
        checked=${d.perFrameAuto} onChange=${(v) => { file.stats = new Map(); setDisplay({ perFrameAuto: v }); }} />` : null}
    <//>
    <${Section} title="On the image">
      <${Toggle} id="show-fits" label="Fit circles" checked=${d.showFits} onChange=${(v) => setDisplay({ showFits: v }, 'overlay')} />
      <${Toggle} id="show-labels" label="Brightness labels" hint="In kpps, next to each fit" checked=${d.showLabels} onChange=${(v) => setDisplay({ showLabels: v }, 'overlay')} />
      <${Toggle} id="color-class" label="Colour fits by aggregation" hint="Monomer, dimer, trimer, multimer from each channel's single-particle peak"
        checked=${d.colorBy === 'class'} onChange=${(v) => setDisplay({ colorBy: v ? 'class' : 'none' }, 'overlay')} />
      <${Toggle} id="show-grid" label="Channel outlines" hint="In the All view" checked=${d.showGrid} onChange=${(v) => setDisplay({ showGrid: v }, 'overlay')} />
      <${Toggle} id="show-scale" label="Scale bar" checked=${d.showScaleBar} onChange=${(v) => setDisplay({ showScaleBar: v }, 'overlay')} />
      <div class="btn-row"><${Btn} icon="save" onClick=${onSaveImage} id="save-image">Save image<//></div>
    <//>
  </div>`;
}

export { channelInfo };
