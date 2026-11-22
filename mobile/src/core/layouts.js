// Microscope presets: how a camera frame divides into emission channels, and
// the pixel size.
//
// Rectangles are in frame pixels, x = column and y = row with row 0 first,
// exactly as PsyFit slices the array (utils.integrate_sif). The image is shown
// with row 0 at the bottom (matplotlib origin='lower'), so for the MCL:
//
//     ┌──────┬───────┐      region 1 Blue  = rows h/2.., cols ..w/2
//     │ Blue │ Green │      region 2 Green = rows h/2.., cols w/2..
//     ├──────┼───────┤      region 3 Red   = rows ..h/2, cols ..w/2
//     │ Red  │  NIR  │      region 4 NIR   = rows ..h/2, cols w/2..
//     └──────┴───────┘

// Channel colours follow PsyFit's "Intuitive, colorblind-safe" merge palette.
const MCL_CHANNELS = [
  { id: 'blue', label: 'Blue', short: 'B', color: '#597AFF', hue: 'blue', region: '1' },
  { id: 'green', label: 'Green', short: 'G', color: '#33FF85', hue: 'green', region: '2' },
  { id: 'red', label: 'Red', short: 'R', color: '#FF2A00', hue: 'red', region: '3' },
  { id: 'nir', label: 'NIR', short: 'NIR', color: '#FF80AC', hue: 'pink', region: '4' },
];

/** Side of PsyFit's fixed "Mr Beam" crop on the Nikon: image[-280:, :280]. */
export const NIKON_BEAM_PX = 280;

/**
 * Ways the Nikon frame can be divided. `lr` and `tb` are the two halves of an
 * image splitter; `beam` is the single region PsyFit's Nikon tools analyse.
 */
export const NIKON_SPLITS = {
  lr: {
    label: 'Left · Right',
    note: 'Two emission channels side by side',
    channels: [
      { id: 'ch1', label: 'Left', short: 'L', color: '#33FF85', hue: 'green' },
      { id: 'ch2', label: 'Right', short: 'R', color: '#FF5CE1', hue: 'magenta' },
    ],
  },
  tb: {
    label: 'Top · Bottom',
    note: 'Two emission channels, one above the other',
    channels: [
      { id: 'ch1', label: 'Top', short: 'T', color: '#33FF85', hue: 'green' },
      { id: 'ch2', label: 'Bottom', short: 'Bt', color: '#FF5CE1', hue: 'magenta' },
    ],
  },
  beam: {
    label: 'Mr Beam crop',
    note: `The ${NIKON_BEAM_PX} px region PsyFit analyses on the Nikon (top left of the frame)`,
    channels: [
      { id: 'ch1', label: 'Mr Beam', short: 'MB', color: '#33FF85', hue: 'green' },
    ],
  },
};
export const NIKON_SPLIT_ORDER = ['lr', 'tb', 'beam'];

export const SCOPES = {
  mcl: {
    id: 'mcl', name: 'MCL', pixUm: 0.1011,
    note: 'Four emission bands, one per quadrant of the camera (60×, 101.1 nm pixels)',
    channels: MCL_CHANNELS,
  },
  nikon: {
    id: 'nikon', name: 'Nikon', pixUm: 0.107, split: 'lr',
    note: '', channels: [],
  },
  full: {
    id: 'full', name: 'Full frame', pixUm: 0.1011,
    note: 'No channel split: the whole frame is one image',
    channels: [],
  },
};
export const SCOPE_ORDER = ['mcl', 'nikon', 'full'];

/** Choose how Nikon frames are divided. Returns true when the split changed. */
export function setNikonSplit(split) {
  const def = NIKON_SPLITS[split];
  if (!def) return false;
  const scope = SCOPES.nikon;
  const changed = scope.split !== split || !scope.channels.length;
  scope.split = split;
  scope.channels = def.channels;
  scope.note = `${def.note} (100×, 107 nm pixels)`;
  return changed;
}
export const nikonSplit = () => SCOPES.nikon.split;
setNikonSplit('lr');

const ALL = { id: 'all', label: 'All', short: 'All', color: '#aab3c5', hue: 'gray' };
const BORDER = { id: 'border', label: 'Border', short: '–', color: '#8d97ab', hue: 'gray' };

/** Label, short label, marker colour and overlay hue of a channel. Never null. */
export function channelInfo(scopeId, channel) {
  if (channel === 'all') return ALL;
  if (channel === 'border') return BORDER;
  const scope = SCOPES[scopeId];
  const hit = scope && scope.channels.find((c) => c.id === channel);
  if (hit) return hit;
  for (const id of SCOPE_ORDER) {
    const other = SCOPES[id].channels.find((c) => c.id === channel);
    if (other) return other;
  }
  return { id: channel, label: String(channel), short: String(channel).slice(0, 3), color: ALL.color, hue: 'gray' };
}

/**
 * Which microscope a file came from, when its name (or the path Solis saved it
 * under) says so. Returns '' when it does not.
 */
export function guessScope(name, info) {
  const text = `${name || ''} ${(info && info.OriginalFilename) || ''}`;
  if (/(^|[^a-z])mcl([^a-z]|$)/i.test(text)) return 'mcl';
  if (/nikon|mr[\s._-]?beam/i.test(text)) return 'nikon';
  return '';
}

/** Objective magnification written in a file name ("..._100x_..."), or null. */
export function magnificationFromName(name) {
  const m = /(?:^|[^0-9a-z.])(\d{2,3})\s?x(?:[^a-z0-9]|$)/i.exec(String(name || ''));
  if (!m) return null;
  const mag = parseInt(m[1], 10);
  return mag >= 10 && mag <= 200 ? mag : null;
}

/**
 * Micrometres per image pixel. MCL: 0.1011 (60×). Nikon: 0.107 at 100×,
 * scaled by the magnification in the file name when there is one. Binned
 * acquisitions have proportionally larger pixels.
 */
export function pixelSize(scopeId, name, info) {
  const scope = SCOPES[scopeId] || SCOPES.full;
  let um = scope.pixUm;
  if (scopeId === 'nikon') {
    const mag = magnificationFromName(name);
    if (mag) um = (0.107 * 100) / mag;
  }
  const bin = info && info.xbin > 1 ? info.xbin : 1;
  return um * bin;
}

/**
 * Rectangle of a channel inside a frame, or null when this microscope has no
 * such channel. `src` needs `width` and `height`.
 * @returns {null | {x: number, y: number, w: number, h: number}}
 */
export function channelRect(scopeId, channel, src) {
  if (!src) return null;
  const W = src.width, H = src.height;
  if (channel === 'all') return { x: 0, y: 0, w: W, h: H };
  const mw = Math.floor(W / 2), mh = Math.floor(H / 2);
  let r = null;
  if (scopeId === 'mcl') {
    if (channel === 'blue') r = { x: 0, y: mh, w: mw, h: H - mh };
    else if (channel === 'green') r = { x: mw, y: mh, w: W - mw, h: H - mh };
    else if (channel === 'red') r = { x: 0, y: 0, w: mw, h: mh };
    else if (channel === 'nir') r = { x: mw, y: 0, w: W - mw, h: mh };
  } else if (scopeId === 'nikon') {
    const split = SCOPES.nikon.split;
    if (split === 'lr') {
      if (channel === 'ch1') r = { x: 0, y: 0, w: mw, h: H };
      else if (channel === 'ch2') r = { x: mw, y: 0, w: W - mw, h: H };
    } else if (split === 'tb') {
      if (channel === 'ch1') r = { x: 0, y: mh, w: W, h: H - mh };      // shown on top
      else if (channel === 'ch2') r = { x: 0, y: 0, w: W, h: mh };
    } else if (channel === 'ch1') {
      const s = Math.min(NIKON_BEAM_PX, W, H);
      r = { x: 0, y: H - s, w: s, h: s };
    }
  }
  return r && r.w >= 8 && r.h >= 8 ? r : null;
}

/** The channels of this microscope that exist in a frame of this size. */
export function availableChannels(scopeId, src) {
  const scope = SCOPES[scopeId];
  if (!scope || !src) return [];
  return scope.channels.filter((c) => channelRect(scopeId, c.id, src));
}

/** Id of the channel that holds frame pixel (x, y), or null when none does. */
export function channelAt(scopeId, src, x, y) {
  for (const c of availableChannels(scopeId, src)) {
    const r = channelRect(scopeId, c.id, src);
    if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return c.id;
  }
  return null;
}

/** Copy a rectangle out of a row-major image of width `sw`. */
export function cropRect(data, sw, rect) {
  const out = new Float32Array(rect.w * rect.h);
  for (let y = 0; y < rect.h; y++) {
    const start = (rect.y + y) * sw + rect.x;
    out.set(data.subarray(start, start + rect.w), y * rect.w);
  }
  return out;
}
