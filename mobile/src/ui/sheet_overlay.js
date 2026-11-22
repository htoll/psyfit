// Overlay panel: layers (any channel of any open file), channel registration
// and colocalization between two layers.

import { html, Btn, IconBtn, Stepper, Toggle, Section, Icon } from './widgets.js';
import { Chart, Legend } from './plots.js';
import { THEME, fmtNum, fmtDate } from './theme.js';
import {
  S, bump, persist, toast, activeFile, fileById, setOverlay, toggleLayer, updateLayer, removeLayer, layerLabel, shortName,
  registerFromPsfs, registerFromAperture, clearRegistration, describeAffine, alignLayer, computeColoc, colocCsv,
} from '../store.js';
import { SCOPES, availableChannels, channelInfo } from '../core/layouts.js';
import { HUE_NAMES, hueCss } from '../core/render.js';
import { saveFile, copyText } from '../platform.js';

function LayerRow({ layer, index, count }) {
  const nudge = (dx, dy) => updateLayer(layer.id, { nudge: [layer.nudge[0] + dx, layer.nudge[1] + dy] });
  const nextHue = () => updateLayer(layer.id, { hue: HUE_NAMES[(HUE_NAMES.indexOf(layer.hue) + 1) % HUE_NAMES.length] });
  const moved = layer.nudge[0] || layer.nudge[1];
  return html`<li class="layer">
    <div class="layer-head">
      <button type="button" class="hue" style=${{ background: hueCss(layer.hue) }} onClick=${nextHue}
        title="Change colour" aria-label=${`Colour ${layer.hue}. Tap to change`}></button>
      <span class="layer-name">${layerLabel(layer)}${index === 0 && count > 1 ? html` <small>base</small>` : ''}</span>
      <${IconBtn} icon=${layer.visible ? 'eye' : 'eyeoff'} title=${layer.visible ? 'Hide layer' : 'Show layer'} onClick=${() => updateLayer(layer.id, { visible: !layer.visible })} size=${18} />
      ${count > 1 ? html`<${IconBtn} icon="close" title="Remove layer" onClick=${() => removeLayer(layer.id)} size=${18} />` : null}
    </div>
    ${index > 0 ? html`<div class="layer-nudge">
      <span class="field-label">Shift ${moved ? `${layer.nudge[0].toFixed(1)}, ${layer.nudge[1].toFixed(1)} px` : 'none'}</span>
      <div class="nudge-pad" role="group" aria-label="Shift this layer">
        <${IconBtn} icon="left" title="Shift left" onClick=${() => nudge(-0.5, 0)} size=${16} />
        <${IconBtn} icon="up" title="Shift up" onClick=${() => nudge(0, 0.5)} size=${16} />
        <${IconBtn} icon="down" title="Shift down" onClick=${() => nudge(0, -0.5)} size=${16} />
        <${IconBtn} icon="right" title="Shift right" onClick=${() => nudge(0.5, 0)} size=${16} />
      </div>
      <${Btn} small=${true} onClick=${() => { const r = alignLayer(layer.id); if (r) toast(`Shifted by ${r.m[2].toFixed(2)}, ${r.m[5].toFixed(2)} px using ${r.n} common PSFs.`); }}>Auto<//>
      ${moved ? html`<${Btn} small=${true} kind="btn-quiet" onClick=${() => updateLayer(layer.id, { nudge: [0, 0] })}>Reset<//>` : null}
    </div>` : null}
  </li>`;
}

function AddLayers() {
  return html`<div class="add-layers">
    ${S.files.map((f) => {
      const chans = availableChannels(f.scope, f.src);
      const on = new Set(S.layers.filter((l) => l.fileId === f.id).map((l) => l.channel));
      const items = chans.length ? chans : [{ id: 'all', label: 'Whole frame', color: channelInfo(f.scope, 'all').color }];
      return html`<div class="add-file" key=${f.id}>
        <span class="add-name">${shortName(f.name, 34)}</span>
        <div class="pill-row">
          ${items.map((c) => html`<button type="button" key=${c.id} class=${`pill${on.has(c.id) ? ' is-on' : ''}`} aria-pressed=${String(on.has(c.id))}
            onClick=${() => toggleLayer(f.id, c.id)}><span class="dot" style=${{ background: c.color }}></span>${c.label}</button>`)}
        </div>
      </div>`;
    })}
  </div>`;
}

function Registration({ file }) {
  const scope = SCOPES[file.scope];
  const reg = S.registration[file.scope];
  if (!scope.channels.length) return html`<p class="hint">Full-frame files have no channels to register. Layers from different files can still be shifted onto each other above.</p>`;
  return html`<div>
    <p class="hint">Channels land on different parts of the camera and never line up exactly. Measure the offsets once per microscope alignment; they are remembered on this device and applied to every ${scope.name} file.</p>
    <div class="btn-row">
      <${Btn} id="reg-psf" onClick=${() => { const r = registerFromPsfs(file); if (r) toast(`Registered to ${channelInfo(file.scope, r.ref).label} from PSFs in this image.`); }}>From PSFs in this image<//>
      <${Btn} id="reg-aperture" onClick=${() => { const r = registerFromAperture(file); if (r) toast('Channels aligned on their aperture centres.'); }}>From the aperture<//>
    </div>
    <p class="hint">“From PSFs” needs particles that show in several channels (an alignment standard is ideal) and solves shift, scale and rotation. “From the aperture” is rougher: it lines up the circular field stop and needs its edge to be visible.</p>
    ${reg ? html`<div class="reg">
      <div class="reg-head"><strong>${reg.method === 'aperture' ? 'Aperture alignment' : 'PSF registration'}</strong>
        <span class="row-meta">${shortName(reg.source || '', 30)} · ${fmtDate(new Date(reg.when || Date.now()).toISOString())}</span></div>
      <ul class="reg-list">
        ${scope.channels.filter((c) => reg.channels[c.id]).map((c) => {
          const e = reg.channels[c.id];
          const d = describeAffine(e.m);
          return html`<li key=${c.id}><span class="dot" style=${{ background: c.color }}></span><span class="reg-name">${c.label}</span>
            <span class="num">${e.kind === 'reference' ? 'reference' : e.kind === 'failed' ? (e.found !== undefined ? `no match (${e.found} PSFs)` : 'not found')
              : `${d.tx >= 0 ? '+' : ''}${d.tx.toFixed(2)}, ${d.ty >= 0 ? '+' : ''}${d.ty.toFixed(2)} px${e.kind === 'affine' ? ` · ×${d.scaleX.toFixed(4)} · ${d.rotationDeg.toFixed(2)}°` : ''}${e.n ? ` · ${e.n} PSFs` : ''}${Number.isFinite(e.rms) && e.rms > 0 ? ` · rms ${e.rms.toFixed(2)}` : ''}`}</span></li>`;
        })}
      </ul>
      <${Toggle} id="use-reg" label="Apply registration" checked=${S.display.useReg} onChange=${(v) => { S.display.useReg = v; persist(); bump('scene'); }} />
      <div class="btn-row">
        <${Btn} small=${true} kind="btn-quiet" onClick=${async () => { const ok = await copyText(JSON.stringify(reg)); toast(ok ? 'Registration copied as JSON.' : 'Copying is not available here.', ok ? 'info' : 'warn'); }}>Copy as JSON<//>
        <${Btn} small=${true} kind="btn-quiet" icon="trash" onClick=${() => clearRegistration(file.scope)}>Forget<//>
      </div>
    </div>` : html`<p class="hint">No registration stored for ${scope.name}: channels are overlaid by their nominal position only.</p>`}
  </div>`;
}

function Coloc() {
  const layers = S.layers.filter((l) => l.visible && fileById(l.fileId));
  if (layers.length < 2) return html`<p class="hint">Show two layers (two channels, or the same channel of two files) to measure how many particles coincide.</p>`;
  const a = layers.find((l) => l.id === S.coloc.a) || layers[0];
  const b = layers.find((l) => l.id === S.coloc.b && l !== a) || layers.find((l) => l !== a);
  const r = S.coloc.result;
  const pick = (which, id) => { S.coloc[which] = id; S.coloc.result = null; bump('overlay'); };
  const pts = r ? r.pairs.map((p) => ({ x: p.a.brightness, y: p.b.brightness, note: `${p.d.toFixed(2)} px apart` })) : [];
  const series = [{ name: 'Colocalized pairs', color: '#9085e9', points: pts, line: false, markers: true }];
  if (r && r.regression && pts.length > 1) {
    const xs = pts.map((p) => p.x), x0 = Math.min(...xs), x1 = Math.max(...xs);
    series.push({ name: 'Linear fit', color: THEME.ink, width: 1.75, dash: [5, 4], hit: false,
      points: [{ x: x0, y: r.regression.intercept + r.regression.slope * x0 }, { x: x1, y: r.regression.intercept + r.regression.slope * x1 }] });
  }
  const exportPairs = async () => {
    try { const res = await saveFile('colocalized_pairs.csv', colocCsv(), 'text/csv'); if (res !== 'declined') toast('Pairs saved as CSV.'); } catch (err) { toast(err.message, 'error'); }
  };
  return html`<div>
    <div class="field-row">
      <label class="field"><span class="field-label">Layer A</span>
        <select value=${a.id} onChange=${(e) => pick('a', e.currentTarget.value)}>${layers.map((l) => html`<option key=${l.id} value=${l.id}>${layerLabel(l)}</option>`)}</select></label>
      <label class="field"><span class="field-label">Layer B</span>
        <select value=${b.id} onChange=${(e) => pick('b', e.currentTarget.value)}>${layers.filter((l) => l !== a).map((l) => html`<option key=${l.id} value=${l.id}>${layerLabel(l)}</option>`)}</select></label>
      <${Stepper} id="coloc-radius" label="Within (px)" value=${S.coloc.radius} min=${0.5} max=${10} step=${0.5} onChange=${(v) => { S.coloc.radius = v; S.coloc.result = null; bump('overlay'); }} />
    </div>
    <div class="btn-row">
      <${Btn} kind="btn-primary" id="run-coloc" onClick=${() => { S.coloc.a = a.id; S.coloc.b = b.id; computeColoc(); }}>Find colocalized PSFs<//>
      ${r ? html`<${Btn} small=${true} icon="save" onClick=${exportPairs}>CSV<//>` : null}
    </div>
    ${r ? html`<div class="coloc-result" role="status">
      <div class="stat-row">
        <div class="stat"><span class="stat-num">${(100 * r.fracA).toFixed(0)}%</span><span class="stat-label">of ${layerLabel(a)} (${r.pairs.length} of ${r.nA})</span></div>
        <div class="stat"><span class="stat-num">${(100 * r.fracB).toFixed(0)}%</span><span class="stat-label">of ${layerLabel(b)} (${r.pairs.length} of ${r.nB})</span></div>
        <div class="stat"><span class="stat-num">${Number.isFinite(r.meanDist) ? r.meanDist.toFixed(2) : '–'}</span><span class="stat-label">px mean separation</span></div>
      </div>
      ${pts.length ? html`<figure class="plot">
        <figcaption><strong>Brightness of the pairs</strong>${r.regression ? html`<span class="num">slope ${fmtNum(r.regression.slope, 3)} · R² ${r.regression.r2.toFixed(3)}</span>` : null}</figcaption>
        <${Legend} items=${series.length > 1 ? [{ name: 'Colocalized pairs', color: '#9085e9' }, { name: 'Linear fit', color: THEME.ink, line: true }] : []} />
        <${Chart} height=${200} spec=${{ aria: 'Brightness in layer B against layer A for colocalized pairs', xLabel: `${layerLabel(a)} (pps)`, yLabel: `${layerLabel(b)} (pps)`, xName: 'A', yName: 'B', xUnit: 'pps', yUnit: 'pps', series, yZero: true, xZero: true }} />
      </figure>` : null}
      <p class="hint">Colocalized fits are circled in bright green on the image. Each particle pairs with at most one partner, nearest first.</p>
    </div>` : null}
  </div>`;
}

export function OverlaySheet() {
  const file = activeFile();
  if (!file) return html`<div class="sheet-body"><p class="hint">Open a file to overlay its channels.</p></div>`;
  const count = S.layers.length;
  return html`<div class="sheet-body">
    <${Section} title="Layers" aside=${html`<${Toggle} id="overlay-on" label="Overlay" checked=${S.overlay} onChange=${(v) => setOverlay(v)} />`}>
      ${S.overlay ? html`<ul class="layers">${S.layers.map((l, i) => html`<${LayerRow} key=${l.id} layer=${l} index=${i} count=${count} />`)}</ul>`
        : html`<p class="hint">Turn Overlay on to show several channels at once, each in its own colour. The first layer is the base that the others are mapped onto.</p>`}
      <h4 class="sub">Add or remove layers</h4>
      <${AddLayers} />
      ${S.overlay && count > 1 ? html`<p class="hint">Set each layer's black and white points under Display. Double-tap a spot to fit it in every layer at once.</p>` : null}
    <//>
    <${Section} title=${`Channel registration · ${SCOPES[file.scope].name}`}>
      <${Registration} file=${file} />
    <//>
    <${Section} title="Colocalization">
      <${Coloc} />
    <//>
  </div>`;
}

export { Icon };
