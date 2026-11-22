// Andor Solis .sif reader (and a minimal writer for demos/tests).
//
// The header walk is a line-for-line port of fujiisoup/sif_parser's `_open`, the
// parser PsyFit's Python tools use, including its token-reading quirks, so both
// agree on where the pixel data starts. Pixel data is little-endian float32,
// one frame after another, row 0 first.
//
// Frames are read lazily with Blob.slice(), so a multi-GB kinetic series costs
// only the frames actually requested.

export const SIF_MAGIC = 'Andor Technology Multi-Channel File\n';

export class SifError extends Error {
  constructor(message) { super(message); this.name = 'SifError'; }
}
/** Thrown when the header runs past the bytes we were given: retry with more. */
export class SifNeedMore extends Error {
  constructor() { super('SIF header extends past the bytes read so far'); this.name = 'SifNeedMore'; }
}

class Cursor {
  constructor(bytes) { this.b = bytes; this.p = 0; }
  tell() { return this.p; }
  seek(p) { this.p = p; }
  byte() {
    if (this.p >= this.b.length) throw new SifNeedMore();
    return this.b[this.p++];
  }
  /** fp.read(n) */
  read(n) {
    if (this.p + n > this.b.length) throw new SifNeedMore();
    const out = this.b.subarray(this.p, this.p + n);
    this.p += n;
    return out;
  }
  readText(n) { return latin1(this.read(n)); }
  /** fp.readline(): through the newline, inclusive. */
  readLine() {
    const start = this.p;
    const b = this.b;
    let i = start;
    while (i < b.length && b[i] !== 10) i++;
    if (i >= b.length) throw new SifNeedMore();
    this.p = i + 1;
    return latin1(b.subarray(start, i + 1));
  }
  /**
   * sif_parser._read_until: a leading terminator is swallowed into the word
   * (harmless to int()/float()), then reads up to the next terminator/newline.
   */
  readUntil(term = ' ') {
    let word = '';
    for (;;) {
      const c = String.fromCharCode(this.byte());
      if (c === term || c === '\n') {
        if (word.length > 0) break;
      }
      word += c;
    }
    return word;
  }
  readInt() { return pyInt(this.readUntil(' ')); }
  readFloat() { return pyFloat(this.readUntil(' ')); }
  /** sif_parser._read_string: a length line, then that many bytes. */
  readString() {
    const n = pyInt(this.readLine());
    return this.readText(n);
  }
  skipSpaces() {
    for (;;) {
      const at = this.p;
      const c = this.byte();
      if (c !== 32 && c !== 10) { this.p = at; return; }
    }
  }
}

function latin1(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + 8192, bytes.length)));
  }
  return s;
}

function pyInt(text) {
  const t = String(text).trim();
  if (!/^[+-]?\d+$/.test(t)) throw new SifError(`Unexpected header token "${t.slice(0, 24)}" (expected an integer)`);
  return parseInt(t, 10);
}

function pyFloat(text) {
  const t = String(text).trim();
  const low = t.toLowerCase();
  if (low === 'nan' || low === '+nan' || low === '-nan') return NaN;
  if (low === 'inf' || low === '+inf' || low === 'infinity') return Infinity;
  if (low === '-inf' || low === '-infinity') return -Infinity;
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) {
    throw new SifError(`Unexpected header token "${t.slice(0, 24)}" (expected a number)`);
  }
  return Number(t);
}

/**
 * Parse a SIF header from the first bytes of a file.
 * @param {Uint8Array} bytes head of the file (throws SifNeedMore if too short)
 * @returns {object} info: sif_parser's keys plus width/height/frames/offset
 */
export function parseSifHeader(bytes) {
  const fp = new Cursor(bytes);
  const info = {};

  if (bytes.length < 36) throw new SifNeedMore();
  if (fp.readText(36) !== SIF_MAGIC) throw new SifError('Not an Andor .sif file (missing "Andor Technology Multi-Channel File" signature)');
  fp.readLine();

  info.SifVersion = pyInt(fp.readUntil(' '));
  fp.readUntil(' '); fp.readUntil(' '); fp.readUntil(' ');
  info.ExperimentTime = fp.readInt();
  info.DetectorTemperature = fp.readFloat();
  fp.read(10);
  fp.readUntil(' ');
  info.ExposureTime = fp.readFloat();
  info.CycleTime = fp.readFloat();
  info.AccumulatedCycleTime = fp.readFloat();
  info.AccumulatedCycles = fp.readInt();
  fp.read(1); fp.read(1);
  info.StackCycleTime = fp.readFloat();
  info.PixelReadoutTime = fp.readFloat();
  fp.readUntil(' '); fp.readUntil(' ');
  info.GainDAC = fp.readFloat();
  fp.readUntil(' '); fp.readUntil(' ');
  info.GateWidth = fp.readFloat();
  for (let i = 0; i < 16; i++) fp.readUntil(' ');
  info.GratingBlaze = fp.readFloat();
  fp.readUntil('\n');

  info.DetectorType = fp.readLine().trim();
  info.DetectorDimensions = [fp.readInt(), fp.readInt()];
  info.OriginalFilename = fp.readString();
  fp.read(2);
  fp.readUntil(' ');
  info.UserText = fp.readString();
  fp.read(1);
  fp.readInt();
  fp.read(8);
  info.ShutterTime = [fp.readFloat(), fp.readFloat()];

  const v = info.SifVersion;
  const secondToken = (line) => (line.trim().split(/\s+/)[1] || '');
  if (v >= 65548 && v <= 65557) {
    for (let i = 0; i < 2; i++) fp.readLine();
  } else if (v === 65558) {
    for (let i = 0; i < 5; i++) fp.readLine();
  } else if (v === 65559 || v === 65564) {
    for (let i = 0; i < 8; i++) fp.readLine();
    info.spectrograph = secondToken(fp.readLine());
  } else if (v === 65565) {
    for (let i = 0; i < 15; i++) fp.readLine();
  } else if (v > 65565) {
    for (let i = 0; i < 8; i++) fp.readLine();
    info.spectrograph = secondToken(fp.readLine());
    fp.readLine();
    for (let i = 0; i < 3; i++) fp.readFloat();
    info.GateGain = fp.readFloat();
    fp.readFloat(); fp.readFloat();
    info.GateDelay = fp.readFloat() * 1e-12;
    info.GateWidth = fp.readFloat() * 1e-12;
    for (let i = 0; i < 8; i++) fp.readLine();
  }
  if (info.spectrograph === undefined) info.spectrograph = 'sif version not checked yet';

  info.SifCalbVersion = pyInt(fp.readUntil(' '));
  if (info.SifCalbVersion === 65540) fp.readLine();
  fp.readLine();            // calibration (or Mechelle pixel calibration)
  fp.readLine();            // old calibration data
  fp.readLine();
  fp.readLine();            // Raman excitation wavelength
  fp.readLine();
  fp.readLine();
  fp.readLine();

  info.FrameAxis = fp.readString();
  info.DataType = fp.readString();
  info.ImageAxis = fp.readString();

  fp.readUntil(' ');
  fp.readUntil(' '); fp.readUntil(' '); fp.readUntil(' '); fp.readUntil(' ');

  const noImages = pyInt(fp.readUntil(' '));
  const noSubimages = pyInt(fp.readUntil(' '));
  info.TotalLength = pyInt(fp.readUntil(' '));
  info.ImageLength = pyInt(fp.readUntil(' '));
  info.NumberOfFrames = noImages;
  info.NumberOfSubImages = noSubimages;
  if (!(noImages > 0) || !(noSubimages > 0)) throw new SifError('SIF header reports no image data');

  let width = 0, height = 0, xbin = 1, ybin = 1;
  for (let i = 0; i < noSubimages; i++) {
    fp.readUntil(' ');
    const area = fp.readLine().trim().split(/\s+/).map(pyInt);
    if (area.length < 6) throw new SifError('Malformed sub-image descriptor in SIF header');
    const [x0, y1, x1, y0] = area;
    ybin = area[4]; xbin = area[5];
    width = Math.trunc((1 + x1 - x0) / xbin);
    height = Math.trunc((1 + y1 - y0) / ybin);
    // Sensor region that was read out (1-based, inclusive; y counts up from the bottom row).
    if (i === 0) info.roi = { left: x0, right: x1, bottom: y0, top: y1 };
  }
  info.xbin = xbin; info.ybin = ybin;
  width = Math.trunc(width);
  height = Math.trunc(height) * noSubimages;
  if (!(width > 0) || !(height > 0)) throw new SifError('SIF header reports an empty image');

  fp.skipSpaces();
  const timestamps = new Array(noImages);
  for (let f = 0; f < noImages; f++) timestamps[f] = pyInt(fp.readLine());
  info.timestamps = timestamps;

  // An optional "0" line (and, in version 65567, a "1" line plus one more number
  // per frame) sits between the time stamps and the pixel data.
  let offset = fp.tell();
  const flag = peekIntLine(fp);
  if (flag !== null) {
    if (flag.value === 0) {
      offset = flag.end;
    } else if (flag.value === 1 && v === 65567) {
      fp.seek(flag.end);
      for (let i = 0; i < noImages; i++) fp.readLine();
      offset = fp.tell();
    }
  }

  info.width = width;
  info.height = height;
  info.frames = noImages;
  info.offset = offset;
  info.frameBytes = width * height * 4;
  return info;
}

/** Python's `int(fp.readline())` inside a try: null when the next line is not a small integer. */
function peekIntLine(fp) {
  const b = fp.b;
  const start = fp.p;
  const limit = Math.min(b.length, start + 24);
  let i = start;
  while (i < limit && b[i] !== 10) i++;
  if (i >= limit) {
    // No newline nearby: binary pixel data. (If we are at the very end of the
    // buffer we cannot tell, so ask for more.)
    if (limit === b.length && b.length - start < 24) throw new SifNeedMore();
    return null;
  }
  const text = latin1(b.subarray(start, i)).trim();
  if (!/^[+-]?\d+$/.test(text)) return null;
  return { value: parseInt(text, 10), end: i + 1 };
}

const HEADER_CHUNKS = [1 << 18, 1 << 22, 1 << 25];

/** @returns {Promise<ArrayBuffer>} */
function blobBytes(blob) {
  if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer();
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error || new Error('File read failed'));
    r.readAsArrayBuffer(blob);
  });
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

function toFloat32(buffer, count) {
  if (LITTLE_ENDIAN) return new Float32Array(buffer, 0, count);
  const view = new DataView(buffer);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** PsyFit's counts -> photons/s scale: counts * (5 / EM gain) / exposure / accumulations. */
export function ppsScale(info, sensitivity = 5.0) {
  const gain = info.GainDAC > 0 ? info.GainDAC : 1;
  const exposure = info.ExposureTime > 0 ? info.ExposureTime : 1;
  const cycles = info.AccumulatedCycles > 0 ? info.AccumulatedCycles : 1;
  return (sensitivity / gain) / exposure / cycles;
}

/**
 * Open a .sif Blob/File. Only the header is read here.
 * @returns {Promise<SifSource>}
 */
export async function openSif(blob, name = (blob && blob.name) || 'untitled.sif') {
  if (!blob || !(blob.size > 0)) throw new SifError('The file is empty');
  let info = null;
  let lastError = null;
  for (const chunk of HEADER_CHUNKS) {
    const head = new Uint8Array(await blobBytes(blob.slice(0, Math.min(chunk, blob.size))));
    try {
      info = parseSifHeader(head);
      break;
    } catch (err) {
      lastError = err;
      if (!(err instanceof SifNeedMore) || head.length >= blob.size) break;
    }
  }
  if (!info) {
    if (lastError instanceof SifNeedMore) throw new SifError('The file ends before its SIF header does (truncated or not a .sif file)');
    throw lastError;
  }
  const available = Math.floor((blob.size - info.offset) / info.frameBytes);
  if (available < 1) throw new SifError('The file ends before the first frame (truncated acquisition?)');
  return new SifSource(blob, name, info, Math.min(available, info.frames));
}

export class SifSource {
  constructor(blob, name, info, frames) {
    this.blob = blob;
    this.name = name;
    this.info = info;
    this.width = info.width;
    this.height = info.height;
    /** Frames actually present in the file (may be fewer than the header claims). */
    this.frames = frames;
    this.truncated = frames < info.frames;
    this.size = blob.size;
  }

  /** Raw counts of frame `i` (0-based) as a fresh Float32Array of width*height. */
  async readFrame(i) {
    if (!(i >= 0 && i < this.frames)) throw new SifError(`Frame ${i + 1} is outside 1-${this.frames}`);
    const { offset, frameBytes } = this.info;
    const start = offset + i * frameBytes;
    const buf = await blobBytes(this.blob.slice(start, start + frameBytes));
    if (buf.byteLength < frameBytes) throw new SifError(`Frame ${i + 1} is incomplete`);
    return toFloat32(buf, this.width * this.height);
  }

  /** Seconds between consecutive frames, best effort (kinetic cycle, else exposure). */
  frameInterval() {
    const i = this.info;
    for (const v of [i.CycleTime, i.AccumulatedCycleTime, i.ExposureTime]) {
      if (v > 0 && Number.isFinite(v)) return v;
    }
    return 1;
  }
}

// ---------------------------------------------------------------------------
// Writer (demo data and tests). Produces a header that both this reader and
// sif_parser walk correctly; field layout follows a Solis 65567-era file.
// ---------------------------------------------------------------------------

function asciiBytes(text) {
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
  return out;
}

/**
 * @param {object} o
 * @param {number} o.width
 * @param {number} o.height
 * @param {Float32Array[]} o.frames raw counts, each width*height, row 0 first
 * @param {number} [o.exposure] seconds
 * @param {number} [o.cycle] kinetic cycle time, seconds
 * @param {number} [o.gain] EM gain (GainDAC)
 * @param {number} [o.accumulations]
 * @param {number} [o.temperature]
 * @param {number} [o.version] SIF version (65559, 65565 or 65567 layouts supported)
 * @returns {Uint8Array}
 */
export function writeSif(o) {
  const {
    width, height, frames,
    exposure = 0.1, cycle = exposure, gain = 300, accumulations = 1,
    temperature = -70, version = 65567, detector = 'DU897_BV',
    originalName = 'C:\\Data\\synthetic.sif', experimentTime = 1700000000,
  } = o;
  const n = frames.length;
  const g = (x) => {
    if (Number.isInteger(x)) return String(x);
    return String(Number(Number(x).toPrecision(7)));
  };
  let h = SIF_MAGIC;
  h += '65538 1\n';
  // Line 3: the tokens sif_parser reads, then filler up to the newline.
  h += `${version} 0 0 1 ${experimentTime} ${g(temperature)} `;
  h += '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00';            // fp.read(10)
  h += `0 ${g(exposure)} ${g(cycle)} ${g(cycle * accumulations)} ${accumulations} `;
  h += '\x00 ';                                                // NULL, space
  h += `${g(cycle * n)} 1e-06 0 1 ${g(gain)} 0 0 0 `;
  for (let i = 0; i < 16; i++) h += '0 ';
  h += '500 0 0 0\n';
  h += `${detector}\n`;
  h += `${width} ${height} ${originalName.length}\n${originalName}`;
  h += ' \n';
  h += '65538 0\n';                                            // user text: length 0
  h += '\n';
  h += '65538 ';
  h += '\x00\x01\x00\x00\x00\x00\x00\x00';                    // fp.read(8)
  h += '0.01 0.01\n';
  const filler = (count) => { for (let i = 0; i < count; i++) h += '0 0 0\n'; };
  if (version >= 65548 && version <= 65557) filler(2);
  else if (version === 65558) filler(5);
  else if (version === 65559 || version === 65564) { filler(8); h += '65540 none\n'; }
  else if (version === 65565) filler(15);
  else if (version > 65565) {
    filler(8);
    h += '65540 none\n';
    h += '0 0\n';
    h += '0 0 0 1 0 0 0 0\n';
    filler(8);
  }
  h += '65539 0 1 0 0\n';       // SifCalbVersion, then the rest is the calibration line
  h += '0 1 0 0\n';
  h += '0 1 0 0\n';
  h += '0\n';
  h += '422\n';
  h += '13\n';
  h += '13\n';
  h += '12\nPixel number' + '6\nCounts' + '12\nPixel number';
  h += `65541 1 ${height} ${width} 1 ${n} 1 ${width * height * n} ${width * height}\n`;
  h += `65538 1 ${height} ${width} 1 1 1 0\n`;
  for (let f = 0; f < n; f++) h += `${Math.round(f * cycle * 1e6)}\n`;
  if (version === 65567) {
    h += '1\n';
    for (let f = 0; f < n; f++) h += `${1000000 + f}\n`;
  } else {
    h += '0\n';
  }
  const head = asciiBytes(h);
  const frameBytes = width * height * 4;
  const out = new Uint8Array(head.length + frameBytes * n + 2);
  out.set(head, 0);
  const view = new DataView(out.buffer);
  let p = head.length;
  for (let f = 0; f < n; f++) {
    const fr = frames[f];
    if (fr.length !== width * height) throw new SifError('writeSif: frame size mismatch');
    for (let i = 0; i < fr.length; i++, p += 4) view.setFloat32(p, fr[i], true);
  }
  out[p] = 48; out[p + 1] = 10;   // trailing "0\n", as real files carry a short footer
  return out;
}
