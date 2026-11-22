// Everything that depends on where the page is running:
//  - as a Claude artifact: Google Drive through the viewer's Drive connector
//    (`claude.use("mcp")`) and file saves through `claude.use("downloads")`;
//  - anywhere else (Safari, a home-screen web app): the iOS Files picker, which
//    lists Google Drive as a location, and the share sheet for exports.
// Plus an on-device cache of opened files (IndexedDB).

import { SIF_MAGIC } from './core/sif.js';
import { calibrationKind, calibrationDate } from './core/spectra.js';

const DRIVE = 'Google Drive';
export const DRIVE_TOOLS = ['search_files', 'download_file_content', 'get_file_metadata', 'list_shared_drives'];
/** Above this size the connector route is not attempted (it returns the whole file in one message). */
export const DRIVE_MAX_BYTES = 48 * 1024 * 1024;

const capCache = new Map();
function cap(name) {
  if (!capCache.has(name)) {
    let p;
    try {
      const c = typeof window !== 'undefined' ? window.claude : null;
      p = c && typeof c.use === 'function' ? Promise.resolve(c.use(name)).catch(() => null) : Promise.resolve(null);
    } catch (e) {
      p = Promise.resolve(null);
    }
    capCache.set(name, p);
  }
  return capCache.get(name);
}

export const inArtifact = () => typeof window !== 'undefined' && !!(window.claude && typeof window.claude.use === 'function');

// ---------------------------------------------------------------------------
// Google Drive (connector)
// ---------------------------------------------------------------------------

export class DriveError extends Error {
  constructor(code, message, retryable) {
    super(message);
    this.name = 'DriveError';
    this.code = code;
    this.retryable = !!retryable;
  }
}

/** What to tell the user for each connector failure. */
export function driveErrorText(err) {
  const code = err && err.code;
  switch (code) {
    case 'needs_reauth': return 'Google Drive needs to be reconnected: claude.ai > Settings > Connectors > Google Drive.';
    case 'server_not_connected':
    case 'selection_required':
    case 'server_not_found': return 'No Google Drive connector is available for your account. Add it in claude.ai > Settings > Connectors, or use "Open from Files" and pick Google Drive there.';
    case 'not_in_manifest':
    case 'consent_required':
    case 'approval_required': return 'Drive access was not allowed for this page. Allow Google Drive in the page\'s Permissions menu, then try again.';
    case 'blocked_by_policy': return 'Your organization blocks this Drive action for pages. Use "Open from Files" and pick Google Drive there.';
    case 'server_unavailable':
    case 'rate_limited': return 'Google Drive did not answer. Check the connection and try again.';
    case 'not_granted':
    case 'capability_disabled':
    case 'capability_removed':
    case 'unavailable': return 'Drive browsing is not available in this view. Use "Open from Files" and pick Google Drive there.';
    case 'too_large': return (err && err.message) || 'That file is too large to fetch through the connector.';
    case 'not_sif': return (err && err.message) || 'Drive returned data that is not a .sif file.';
    case 'no_content': return (err && err.message) || 'Drive returned no content for that file.';
    case 'tool_error': return `Google Drive reported an error: ${(err && err.message) || 'unknown'}`;
    default: return `Google Drive request failed (${code || 'error'}): ${(err && err.message) || 'no details'}`;
  }
}

async function driveCall(tool, input, options) {
  const mcp = await cap('mcp');
  if (!mcp) throw new DriveError('unavailable', 'Drive connector not available in this view');
  const attempt = () => mcp.callTool(DRIVE, tool, input, options);
  try {
    return await attempt();
  } catch (err) {
    const e = err || {};
    if (e.retryable && tool !== 'download_file_content') {
      await new Promise((r) => setTimeout(r, Math.min(e.retryAfterMs || 900 + Math.random() * 600, 8000)));
      try { return await attempt(); } catch (err2) { throw new DriveError((err2 && err2.code) || 'upstream_error', (err2 && err2.message) || String(err2)); }
    }
    throw new DriveError(e.code || 'upstream_error', e.message || String(err), e.retryable);
  }
}

function payloadOf(result) {
  if (!result) return null;
  if (result.payload && typeof result.payload === 'object') return result.payload;
  if (result.structuredContent && typeof result.structuredContent === 'object') return result.structuredContent;
  for (const b of result.content || []) {
    if (b && b.type === 'text' && typeof b.text === 'string') {
      try { const j = JSON.parse(b.text); if (j && typeof j === 'object') return j; } catch (e) { /* not JSON */ }
    }
  }
  if (typeof result.payload === 'string') {
    try { const j = JSON.parse(result.payload); if (j && typeof j === 'object') return j; } catch (e) { /* not JSON */ }
  }
  return null;
}

const q = (text) => String(text).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
const FOLDER = 'application/vnd.google-apps.folder';

function normalizeFile(f) {
  return {
    id: f.id,
    name: f.title || f.name || '(untitled)',
    size: f.fileSize !== undefined ? Number(f.fileSize) : (f.size !== undefined ? Number(f.size) : NaN),
    modified: f.modifiedTime || f.createdTime || '',
    parentId: f.parentId || (Array.isArray(f.parents) ? f.parents[0] : '') || '',
    isFolder: f.mimeType === FOLDER,
    isSif: /\.sif$/i.test(f.title || f.name || '') || f.fileExtension === 'sif',
  };
}

export const drive = {
  /** Resolves true when the Drive connector can be called from this page. */
  async available() {
    return !!(await cap('mcp'));
  },

  /**
   * List .sif files (and, inside a folder, its sub-folders).
   * @param {{text?: string, folderId?: string, pageToken?: string}} o
   * @returns {Promise<{items: object[], nextPageToken: string|null}>}
   */
  async list(o = {}) {
    const terms = String(o.text || '').trim().split(/\s+/).filter(Boolean);
    let query;
    if (o.folderId) {
      query = `parentId = '${q(o.folderId)}' and (mimeType = '${FOLDER}' or title contains '.sif')`;
    } else {
      query = "title contains '.sif'";
    }
    for (const t of terms) query += ` and title contains '${q(t)}'`;
    // Drive often answers with pages of ten or fewer whatever is asked for, so
    // gather a few of them before showing the list.
    const items = [];
    let token = o.pageToken || null;
    for (let page = 0; page < 4; page++) {
      const input = { query, pageSize: 50, excludeContentSnippets: true, orderBy: 'modifiedTime desc' };
      if (token) input.pageToken = token;
      const res = await driveCall('search_files', input, { cache: { staleTime: 20000, gcTime: 300000 } });
      const p = payloadOf(res) || {};
      const files = Array.isArray(p.files) ? p.files : (Array.isArray(p) ? p : []);
      items.push(...files.map(normalizeFile).filter((f) => f.id && (f.isFolder || f.isSif)));
      token = p.nextPageToken || null;
      if (!token || items.length >= 24) break;
    }
    items.sort((a, b) => (Number(b.isFolder) - Number(a.isFolder)) || 0);
    return { items, nextPageToken: token };
  },

  async sharedDrives() {
    const res = await driveCall('list_shared_drives', { pageSize: 50 }, { cache: { staleTime: 300000, gcTime: 600000 } });
    const p = payloadOf(res) || {};
    const list = p.sharedDrives || p.drives || p.items || [];
    return list.map((d) => ({ id: d.id || d.sharedDriveId, name: d.name || d.title || 'Shared drive' })).filter((d) => d.id);
  },

  async info(fileId) {
    const res = await driveCall('get_file_metadata', { fileId, excludeContentSnippets: true }, { cache: { staleTime: 60000, gcTime: 600000 } });
    const p = payloadOf(res);
    return p ? normalizeFile(p) : null;
  },

  /** Download a file's bytes, whatever they are. */
  async downloadRaw(file) {
    const res = await driveCall('download_file_content', { fileId: file.id }, { cache: false });
    const b64 = extractBase64(res);
    if (!b64) throw new DriveError('no_content', `Drive returned no content for ${file.name}.`);
    return decodeBase64(b64);
  },

  /** Download a .sif file's bytes. Throws DriveError('not_sif') if they are not an Andor file. */
  async download(file) {
    if (file.size > DRIVE_MAX_BYTES) {
      throw new DriveError('too_large', `${file.name} is ${(file.size / 1048576).toFixed(0)} MB. The Drive connector hands over whole files, so movies this size have to be opened with "Open from Files" (Google Drive appears there as a location).`);
    }
    const bytes = await this.downloadRaw(file);
    if (!looksLikeSif(bytes)) throw new DriveError('not_sif', 'Drive returned data that is not an Andor .sif file.');
    return bytes;
  },

  /**
   * Spectral calibrations in Drive, newest first: pairs of a
   * "saving_info_<date>_no_tracking.pkl" grid and its "<date>_fits.pkl".
   * @returns {Promise<Array<{date: string, grid: object, fits: object}>>}
   */
  async calibrations() {
    const find = async (query) => {
      const out = [];
      let token = null;
      for (let page = 0; page < 4; page++) {
        const input = { query, pageSize: 50, excludeContentSnippets: true, orderBy: 'modifiedTime desc' };
        if (token) input.pageToken = token;
        const p = payloadOf(await driveCall('search_files', input, { cache: { staleTime: 60000, gcTime: 300000 } })) || {};
        out.push(...(Array.isArray(p.files) ? p.files : []).map(normalizeFile).filter((f) => f.id && !f.isFolder && /\.pkl$/i.test(f.name)));
        token = p.nextPageToken || null;
        if (!token) break;
      }
      return out;
    };
    const grids = await find("title contains 'saving_info' and title contains 'no_tracking'");
    const fits = await find("title contains 'fits' and title contains 'pkl'");
    const pairs = [];
    for (const g of grids) {
      const date = calibrationDate(g.name);
      if (calibrationKind(g.name) !== 'grid' || !date || g.size > CAL_MAX_BYTES || pairs.some((x) => x.date === date)) continue;
      const mine = fits.filter((f) => calibrationKind(f.name) === 'fits' && calibrationDate(f.name) === date);
      if (!mine.length) continue;
      mine.sort((a, b) => (Number(b.name === `${date}_fits.pkl`) - Number(a.name === `${date}_fits.pkl`)) || (a.name.length - b.name.length));
      pairs.push({ date, grid: g, fits: mine[0] });
    }
    return pairs.sort((a, b) => (a.date < b.date ? 1 : -1));
  },

  async permissionState() {
    try {
      const perms = await cap('permissions');
      if (!perms) return 'unknown';
      return await perms.state(`mcp:${DRIVE}`);
    } catch (e) { return 'unknown'; }
  },

  async managePermissions() {
    try { const perms = await cap('permissions'); if (perms) await perms.manage(); } catch (e) { /* panel did not open */ }
  },
};

function looksLikeSif(bytes) {
  if (bytes.length < SIF_MAGIC.length) return false;
  for (let i = 0; i < SIF_MAGIC.length - 1; i++) if (bytes[i] !== SIF_MAGIC.charCodeAt(i)) return false;
  return true;
}

/** Calibration files are a few hundred kB; the full-size "saving_info" files (hundreds of MB) are not for this. */
export const CAL_MAX_BYTES = 8 * 1024 * 1024;

const PLACEHOLDER = /^\(file sent separately\)$/i;
const B64_KEYS = ['blob', 'base64Content', 'content', 'data', 'contents', 'base64', 'bytes'];

/**
 * Find the file's base64 in a download_file_content result. The connector sends
 * the bytes as an MCP embedded resource (content[i].resource.blob) next to a
 * small JSON stub; older shapes put the base64 in a `content` field instead.
 */
export function extractBase64(result) {
  if (!result) return null;
  const blocks = Array.isArray(result.content) ? result.content : [];
  for (const b of blocks) {
    if (b && b.resource && typeof b.resource.blob === 'string' && b.resource.blob.length > 16) return b.resource.blob;
  }
  for (const b of blocks) {
    if (b && typeof b.blob === 'string' && b.blob.length > 16) return b.blob;
    if (b && b.type !== 'text' && typeof b.data === 'string' && b.data.length > 16) return b.data;
  }
  const fromObject = (obj) => {
    if (!obj || typeof obj !== 'object') return null;
    for (const key of B64_KEYS) {
      const v = obj[key];
      if (typeof v === 'string' && v.length > 64 && !PLACEHOLDER.test(v.trim())) return v;
      if (v && typeof v === 'object') { const inner = fromObject(v); if (inner) return inner; }
    }
    if (obj.resource) return fromObject(obj.resource);
    return null;
  };
  let found = fromObject(result.payload) || fromObject(result.structuredContent);
  if (found) return found;
  for (const b of blocks) {
    if (b && typeof b.text === 'string') {
      try { found = fromObject(JSON.parse(b.text)); if (found) return found; } catch (e) { /* not JSON */ }
    }
  }
  // Last resort: the longest base64-looking string anywhere in the result.
  let best = null;
  const walk = (v, depth) => {
    if (depth > 6 || v === null || v === undefined) return;
    if (typeof v === 'string') {
      if (v.length > 256 && (!best || v.length > best.length) && /^[A-Za-z0-9+/_\-\s]+=*\s*$/.test(v.slice(0, 512))) best = v;
    } else if (Array.isArray(v)) v.forEach((x) => walk(x, depth + 1));
    else if (typeof v === 'object') Object.keys(v).forEach((k) => walk(v[k], depth + 1));
  };
  walk(result, 0);
  return best;
}

export function decodeBase64(text) {
  let s = String(text);
  if (/[\s\-_]/.test(s)) s = s.replace(/\s+/g, '').replace(/-/g, '+').replace(/_/g, '/');
  while (s.length % 4) s += '=';
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ---------------------------------------------------------------------------
// Saving / sharing exports
// ---------------------------------------------------------------------------

/**
 * Hand a generated file to the user: the artifact download prompt, else the
 * share sheet (Save to Files, AirDrop...), else a normal browser download.
 * @returns {Promise<'saved'|'shared'|'downloaded'|'declined'>}
 */
export async function saveFile(filename, data, mime = 'application/octet-stream') {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
  const downloads = await cap('downloads');
  if (downloads) {
    try {
      await downloads.save({ filename, data: blob });
      return 'saved';
    } catch (err) {
      if (err && err.code === 'declined') return 'declined';
      if (!err || !['unavailable', 'not_granted', 'capability_disabled', 'capability_removed'].includes(err.code)) {
        throw new Error(err && err.code === 'rejected_extension' ? `This view cannot save .${filename.split('.').pop()} files.` : (err && err.message) || 'The file could not be saved.');
      }
    }
  }
  // Inside an artifact the frame blocks every other way of handing over a file.
  if (inArtifact()) throw new Error('This view cannot save files. Open the page in claude.ai and try again.');
  try {
    const file = new File([blob], filename, { type: mime });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: filename });
      return 'shared';
    }
  } catch (err) {
    if (err && err.name === 'AbortError') return 'declined';
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  return 'downloaded';
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) { return false; }
}

// ---------------------------------------------------------------------------
// On-device cache of opened files (so they reopen instantly and offline)
// ---------------------------------------------------------------------------

const DB_NAME = 'psyfit-mobile', STORE = 'files';
export const CACHE_MAX_FILE = 40 * 1024 * 1024;
const CACHE_MAX_TOTAL = 320 * 1024 * 1024;
const CACHE_MAX_COUNT = 60;

function openDb() {
  return new Promise((resolve, reject) => {
    let req;
    try { req = indexedDB.open(DB_NAME, 1); } catch (e) { reject(e); return; }
    req.onupgradeneeded = () => { req.result.createObjectStore(STORE, { keyPath: 'key' }); };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('cache blocked'));
  });
}

function tx(mode, fn) {
  return openDb().then((db) => new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let result;
    Promise.resolve(fn(store)).then((r) => { result = r; }).catch(reject);
    t.oncomplete = () => { db.close(); resolve(result); };
    t.onerror = () => { db.close(); reject(t.error); };
    t.onabort = () => { db.close(); reject(t.error || new Error('cache write aborted')); };
  }));
}

const reqP = (req) => new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });

export const fileCache = {
  /** Metadata of cached files, most recently opened first (no pixel data). */
  async list() {
    try {
      const rows = await tx('readonly', (s) => reqP(s.getAll()));
      return (rows || []).map((r) => ({ key: r.key, name: r.name, size: r.size, opened: r.opened, origin: r.origin }))
        .sort((a, b) => b.opened - a.opened);
    } catch (e) { return []; }
  },
  async get(key) {
    try {
      const row = await tx('readonly', (s) => reqP(s.get(key)));
      if (!row) return null;
      tx('readwrite', (s) => { row.opened = Date.now(); s.put(row); }).catch(() => {});
      return row;
    } catch (e) { return null; }
  },
  /** Store a file's bytes; silently skipped when too large or storage is unavailable. */
  async put(key, name, bytes, origin) {
    if (!bytes || bytes.byteLength > CACHE_MAX_FILE) return false;
    try {
      const buf = bytes.buffer.byteLength === bytes.byteLength ? bytes.buffer : bytes.slice().buffer;
      await tx('readwrite', (s) => { s.put({ key, name, size: bytes.byteLength, opened: Date.now(), origin, buf }); });
      this.trim().catch(() => {});
      return true;
    } catch (e) { return false; }
  },
  async remove(key) {
    try { await tx('readwrite', (s) => { s.delete(key); }); } catch (e) { /* ignore */ }
  },
  async clear() {
    try { await tx('readwrite', (s) => { s.clear(); }); } catch (e) { /* ignore */ }
  },
  async trim() {
    const rows = await this.list();
    let total = 0;
    const drop = [];
    rows.forEach((r, i) => {
      total += r.size;
      if (i >= CACHE_MAX_COUNT || total > CACHE_MAX_TOTAL) drop.push(r.key);
    });
    if (drop.length) await tx('readwrite', (s) => { drop.forEach((k) => s.delete(k)); });
  },
};

// ---------------------------------------------------------------------------
// Small persisted preferences
// ---------------------------------------------------------------------------

const PREF_KEY = 'psyfit-mobile:v1';
export function loadPrefs() {
  try { const raw = localStorage.getItem(PREF_KEY); return raw ? JSON.parse(raw) : {}; } catch (e) { return {}; }
}
export function savePrefs(obj) {
  try { localStorage.setItem(PREF_KEY, JSON.stringify(obj)); } catch (e) { /* storage unavailable */ }
}

// ---------------------------------------------------------------------------
// Spectral calibrations remembered on this device (small JSON, newest first)
// ---------------------------------------------------------------------------

const CAL_KEY = 'psyfit-mobile:cal:v1';
const CAL_KEEP = 4;
export const calStore = {
  all() {
    try { const raw = localStorage.getItem(CAL_KEY); const list = raw ? JSON.parse(raw) : []; return Array.isArray(list) ? list : []; } catch (e) { return []; }
  },
  save(json) {
    try {
      const rest = this.all().filter((c) => !(c.date === json.date && c.name === json.name));
      const list = [json, ...rest].sort((a, b) => (String(a.date) < String(b.date) ? 1 : -1)).slice(0, CAL_KEEP);
      localStorage.setItem(CAL_KEY, JSON.stringify(list));
      return true;
    } catch (e) { return false; }
  },
  remove(date, name) {
    try { localStorage.setItem(CAL_KEY, JSON.stringify(this.all().filter((c) => !(c.date === date && c.name === name)))); } catch (e) { /* storage unavailable */ }
  },
};
