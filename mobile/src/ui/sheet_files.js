// Files panel: open from the system picker or Google Drive, switch between open
// files, reopen files remembered on this device.

import { useEffect, useRef, useState } from 'preact/hooks';
import { html, Btn, IconBtn, Icon, Section, Facts } from './widgets.js';
import { fmtBytes, fmtDate, fmtNum } from './theme.js';
import {
  S, bump, toast, activeFile, setActive, removeFile, addPickedFiles, addBytes, openCached, loadDemo, setSheet,
} from '../store.js';
import { drive, driveErrorText, fileCache, inArtifact, DRIVE_MAX_BYTES } from '../platform.js';
import { SCOPES } from '../core/layouts.js';

function fileFacts(f) {
  const i = f.src.info;
  const when = i.ExperimentTime > 0 ? new Date(i.ExperimentTime * 1000) : null;
  const gain = i.GainDAC > 0 ? i.GainDAC : 1;
  const scale = (5 / gain) / (i.ExposureTime || 1) / (i.AccumulatedCycles || 1);
  return [
    ['Image', `${f.src.width} × ${f.src.height} px${i.xbin > 1 || i.ybin > 1 ? ` (binned ${i.xbin}×${i.ybin})` : ''}`],
    ['Frames', f.src.truncated ? `${f.src.frames} of ${i.frames} (file ends early)` : String(f.src.frames)],
    ['Exposure', `${fmtNum(i.ExposureTime * 1000, 4)} ms`],
    ['Accumulations', String(i.AccumulatedCycles)],
    f.isMovie ? ['Cycle time', `${fmtNum(i.CycleTime * 1000, 4)} ms`] : null,
    ['EM gain', String(i.GainDAC)],
    ['Counts → pps', `× ${fmtNum(scale, 4)}`],
    ['Sensor', `${i.DetectorType || 'unknown'}, ${fmtNum(i.DetectorTemperature, 3)} °C`],
    ['Pixel size', `${fmtNum(f.pixUm * 1000, 4)} nm (${SCOPES[f.scope].name})`],
    when ? ['Acquired', when.toLocaleString()] : null,
    ['File size', fmtBytes(f.src.size)],
    i.OriginalFilename ? ['Saved as', i.OriginalFilename] : null,
  ];
}

function OpenFiles() {
  const [details, setDetails] = useState(null);
  if (!S.files.length) return null;
  return html`<${Section} title="Open files">
    <ul class="list">
      ${S.files.map((f) => {
        const i = f.src.info;
        const on = f.id === S.activeId;
        return html`<li key=${f.id} class=${`row${on ? ' is-on' : ''}`}>
          <button type="button" class="row-main" onClick=${() => setActive(f.id)} aria-current=${on ? 'true' : undefined}>
            <${Icon} name=${f.isMovie ? 'movie' : 'image'} size=${18} />
            <span class="row-text">
              <span class="row-title">${f.name}</span>
              <span class="row-meta">${f.origin.kind === 'demo' ? 'Simulated demo · ' : ''}${SCOPES[f.scope].name} · ${f.isMovie ? `${f.src.frames} frames · ` : ''}${fmtNum(i.ExposureTime * 1000, 3)} ms${i.AccumulatedCycles > 1 ? ` × ${i.AccumulatedCycles}` : ''} · gain ${i.GainDAC}</span>
            </span>
          </button>
          <${IconBtn} icon="info" title="File details" active=${details === f.id} onClick=${() => setDetails(details === f.id ? null : f.id)} size=${18} />
          <${IconBtn} icon="close" title=${`Close ${f.name}`} onClick=${() => removeFile(f.id)} size=${18} />
          ${details === f.id ? html`<div class="row-detail"><${Facts} rows=${fileFacts(f)} /></div>` : null}
        </li>`;
      })}
    </ul>
  <//>`;
}

function DriveBrowser() {
  const [st, setSt] = useState({ mode: 'recent', folder: null, items: [], next: null, loading: false, error: null, text: '', trail: [] });
  const [picked, setPicked] = useState({});
  const reqId = useRef(0);

  const load = async (patch, append) => {
    const next = { ...st, ...patch, loading: true, error: null };
    if (!append) { next.items = []; next.next = null; }
    setSt(next);
    const my = ++reqId.current;
    try {
      let res;
      if (next.mode === 'shared' && !next.folder) {
        const drives = await drive.sharedDrives();
        res = { items: drives.map((d) => ({ id: d.id, name: d.name, isFolder: true, size: NaN, modified: '' })), nextPageToken: null };
      } else {
        res = await drive.list({ text: next.text, folderId: next.folder ? next.folder.id : (next.mode === 'mine' ? 'root' : undefined), pageToken: append ? st.next : undefined });
      }
      if (my !== reqId.current) return;
      setSt({ ...next, loading: false, items: append ? st.items.concat(res.items) : res.items, next: res.nextPageToken });
    } catch (err) {
      if (my !== reqId.current) return;
      setSt({ ...next, loading: false, error: err });
    }
  };

  useEffect(() => { load({}); }, []);

  const enter = (item) => load({ folder: { id: item.id, name: item.name }, trail: st.trail.concat(st.folder ? [st.folder] : []), text: '' });
  const up = () => {
    const trail = st.trail.slice();
    const folder = trail.pop() || null;
    load({ folder, trail });
  };
  const containing = async (item) => {
    if (!item.parentId) return;
    let name = 'Folder';
    try { const info = await drive.info(item.parentId); if (info) name = info.name; } catch (e) { /* keep generic name */ }
    load({ mode: 'folder', folder: { id: item.parentId, name }, trail: [], text: '' });
  };

  const chosen = st.items.filter((it) => picked[it.id]);
  const openChosen = async () => {
    const list = chosen.slice();
    if (!list.length) return;
    const job = { cancelled: false };
    S.busy = { label: 'Fetching from Google Drive', done: 0, total: list.length, cancel: () => { job.cancelled = true; } };
    bump();
    let ok = 0;
    for (let i = 0; i < list.length; i++) {
      if (job.cancelled) break;
      const it = list[i];
      const key = `drive:${it.id}:${it.modified}`;
      try {
        const cached = await fileCache.get(key);
        const bytes = cached ? new Uint8Array(cached.buf) : await drive.download(it);
        S.busy = null;
        ok += await addBytes(it.name, bytes, { kind: 'drive', id: it.id }, cached ? null : key);
        if (i < list.length - 1) { S.busy = { label: 'Fetching from Google Drive', done: i + 1, total: list.length, cancel: () => { job.cancelled = true; } }; bump(); }
      } catch (err) {
        toast(`${it.name}: ${driveErrorText(err)}`, 'error');
      }
    }
    S.busy = null;
    setPicked({});
    if (ok) setSheet(null); else bump();
  };

  const tabs = [['recent', 'Recent'], ['mine', 'My Drive'], ['shared', 'Shared drives']];
  return html`<div class="drive">
    <form class="search" onSubmit=${(e) => { e.preventDefault(); load({}); }}>
      <label class="visually-hidden" for="drive-search">Search .sif files in Drive</label>
      <input id="drive-search" type="search" placeholder="Search .sif names" value=${st.text} enterkeyhint="search"
        onInput=${(e) => setSt({ ...st, text: e.currentTarget.value })} />
      <${Btn} icon="search" kind="btn-quiet" title="Search" onClick=${() => load({})} />
    </form>
    <div class="pill-row" role="group" aria-label="Where to look">
      ${tabs.map(([mode, label]) => html`<button type="button" key=${mode} class=${`pill${st.mode === mode && !st.folder ? ' is-on' : ''}`}
        onClick=${() => load({ mode, folder: null, trail: [], text: mode === 'recent' ? st.text : '' })}>${label}</button>`)}
    </div>
    ${st.folder ? html`<div class="crumb"><${IconBtn} icon="up" title="Up one folder" onClick=${up} size=${18} /><${Icon} name="folder" size=${16} /><span>${st.folder.name}</span></div>` : null}
    ${st.error ? html`<div class="note is-error" role="alert">
        <p>${driveErrorText(st.error)}</p>
        <div class="btn-row">
          <${Btn} small=${true} onClick=${() => load({})}>Try again<//>
          ${['not_in_manifest', 'consent_required', 'approval_required'].includes(st.error.code) ? html`<${Btn} small=${true} onClick=${() => drive.managePermissions()}>Open permissions<//>` : null}
        </div>
      </div>` : null}
    <ul class="list drive-list" aria-busy=${st.loading ? 'true' : 'false'}>
      ${st.items.map((it) => {
        const tooBig = !it.isFolder && it.size > DRIVE_MAX_BYTES;
        return html`<li key=${it.id} class=${`row${picked[it.id] ? ' is-on' : ''}`}>
          <button type="button" class="row-main" disabled=${tooBig}
            onClick=${() => (it.isFolder ? enter(it) : setPicked({ ...picked, [it.id]: !picked[it.id] }))}
            aria-pressed=${it.isFolder ? undefined : String(!!picked[it.id])}>
            ${it.isFolder ? html`<${Icon} name="folder" size=${18} />` : html`<span class=${`check${picked[it.id] ? ' is-on' : ''}`}>${picked[it.id] ? html`<${Icon} name="check" size=${14} />` : null}</span>`}
            <span class="row-text">
              <span class="row-title">${it.name}</span>
              <span class="row-meta">${it.isFolder ? 'Folder' : `${fmtDate(it.modified)} · ${fmtBytes(it.size)}${tooBig ? ' · movie: open it with “Open from Files”' : ''}`}</span>
            </span>
          </button>
          ${!it.isFolder && it.parentId && !st.folder ? html`<${IconBtn} icon="folder" title="Show the folder this file is in" onClick=${() => containing(it)} size=${18} />` : null}
        </li>`;
      })}
    </ul>
    ${st.loading ? html`<p class="hint" role="status">Asking Google Drive…</p>` : null}
    ${!st.loading && !st.error && !st.items.length ? html`<p class="hint">No .sif files here.</p>` : null}
    <div class="btn-row">
      ${st.next && !st.loading ? html`<${Btn} small=${true} onClick=${() => load({}, true)}>More results<//>` : null}
      ${chosen.length ? html`<${Btn} kind="btn-primary" onClick=${openChosen} id="drive-open">Open ${chosen.length} ${chosen.length === 1 ? 'file' : 'files'}<//>` : null}
    </div>
  </div>`;
}

export function FilesSheet() {
  const inputRef = useRef(null);
  const [driveOk, setDriveOk] = useState(null);
  const [cached, setCached] = useState([]);
  useEffect(() => {
    let alive = true;
    drive.available().then((ok) => { if (alive) setDriveOk(ok); });
    fileCache.list().then((rows) => { if (alive) setCached(rows); });
    return () => { alive = false; };
  }, [S.files.length]);

  const openNames = new Set(S.files.map((f) => f.name));
  const remembered = cached.filter((c) => !openNames.has(c.name)).slice(0, 12);
  const demo = async (kind) => { if (await loadDemo(kind)) setSheet(null); };

  return html`<div class="sheet-body">
    <${Section}>
      <input ref=${inputRef} class="visually-hidden" id="file-input" type="file" multiple tabindex="-1" aria-hidden="true"
        onChange=${async (e) => { const files = e.currentTarget.files; const n = await addPickedFiles(files); e.currentTarget.value = ''; if (n) setSheet(null); }} />
      <div class="btn-row">
        <${Btn} kind="btn-primary" icon="files" onClick=${() => inputRef.current && inputRef.current.click()} id="open-files">Open from Files<//>
        <${Btn} onClick=${() => demo('image')} id="demo-image">Demo image<//>
        <${Btn} onClick=${() => demo('movie')} id="demo-movie">Demo movie<//>
      </div>
      <p class="hint">On iPhone the picker lists Google Drive under Browse once the Drive app is installed. Movies of any size open this way: only the frames you ask for are read.</p>
    <//>
    <${OpenFiles} />
    <${Section} title="Google Drive">
      ${driveOk === null ? html`<p class="hint" role="status">Checking for the Drive connector…</p>` : null}
      ${driveOk === true ? html`<${DriveBrowser} />` : null}
      ${driveOk === false ? html`<p class="hint">${inArtifact()
        ? 'The Google Drive connector is not available in this view. Open the page in claude.ai with Google Drive connected, or use “Open from Files” above.'
        : 'Browsing Drive in the app needs the page to run as a Claude artifact with the Google Drive connector. Here, use “Open from Files” above: Google Drive is one of its locations.'}</p>` : null}
    <//>
    ${remembered.length ? html`<${Section} title="On this device" aside=${html`<${Btn} small=${true} kind="btn-quiet" onClick=${async () => { await fileCache.clear(); setCached([]); }}>Forget all<//>`}>
      <ul class="list">
        ${remembered.map((c) => html`<li key=${c.key} class="row">
          <button type="button" class="row-main" onClick=${async () => { if (await openCached(c.key)) setSheet(null); }}>
            <${Icon} name="image" size=${18} />
            <span class="row-text"><span class="row-title">${c.name}</span>
              <span class="row-meta">${c.origin || 'Files'} · ${fmtBytes(c.size)} · opened ${fmtDate(new Date(c.opened).toISOString())}</span></span>
          </button>
        </li>`)}
      </ul>
    <//>` : null}
  </div>`;
}

export { activeFile };
