// A small reader for Python pickles, enough for PsyFit's spectral calibration
// files: dicts, lists, tuples, strings, numbers, None/bools, and NumPy arrays
// and scalars. Nothing is executed: any other object in the stream is refused.
//
// Result types: dict -> Map, list and tuple -> Array, str -> string,
// bytes -> Uint8Array, int/float/NumPy scalar -> number, None -> null,
// NumPy array -> Float64Array with a `shape` property.

export class PickleError extends Error {
  constructor(message) { super(message); this.name = 'PickleError'; }
}

const MARK = Symbol('mark');
const utf8 = new TextDecoder('utf-8');
const latin1 = (b) => { let s = ''; for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]); return s; };

class Global {
  constructor(module, name) { this.module = module; this.name = name; this.key = `${module}.${name}`; }
}
class DType {
  constructor(code) { this.code = String(code); this.order = '='; }
}
/** A NumPy array between its creation and the BUILD that fills it (it can be memoized in between). */
class ArrayStub { constructor() { this.array = null; } }

const NUMPY = /^numpy\.(_?core\.)?(multiarray\.)?/;
const globalKind = (g) => (g instanceof Global && NUMPY.test(g.key) ? g.key.replace(NUMPY, '') : null);

/** Read one element of a NumPy dtype from a DataView. */
function readElement(view, at, dtype) {
  const little = dtype.order !== '>';
  switch (dtype.code.replace(/^[<>=|]/, '')) {
    case 'f8': return view.getFloat64(at, little);
    case 'f4': return view.getFloat32(at, little);
    case 'i8': return Number(view.getBigInt64(at, little));
    case 'u8': return Number(view.getBigUint64(at, little));
    case 'i4': return view.getInt32(at, little);
    case 'u4': return view.getUint32(at, little);
    case 'i2': return view.getInt16(at, little);
    case 'u2': return view.getUint16(at, little);
    case 'i1': return view.getInt8(at);
    case 'u1': case 'b1': return view.getUint8(at);
    default: throw new PickleError(`NumPy type "${dtype.code}" is not supported`);
  }
}
const itemSize = (dtype) => {
  const n = parseInt(dtype.code.replace(/^[<>=|]?[a-z]/i, ''), 10);
  if (!(n > 0)) throw new PickleError(`NumPy type "${dtype.code}" is not supported`);
  return n;
};
const asBytes = (v) => (v instanceof Uint8Array ? v : Uint8Array.from(String(v), (c) => c.charCodeAt(0) & 0xff));

function decodeArray(shape, dtype, raw) {
  const bytes = asBytes(raw), size = itemSize(dtype);
  const count = shape.reduce((a, b) => a * b, 1);
  if (count * size > bytes.length) throw new PickleError('NumPy array data is shorter than its shape');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = readElement(view, i * size, dtype);
  out.shape = shape.slice();
  return out;
}

function reduce(callable, args) {
  const kind = globalKind(callable);
  if (kind === 'dtype') return new DType(args[0]);
  if (kind === 'scalar') {
    const bytes = asBytes(args[1]);
    return readElement(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), 0, args[0]);
  }
  if (kind === '_reconstruct') return new ArrayStub();
  const name = callable instanceof Global ? callable.key : typeof callable;
  throw new PickleError(`This file contains a Python object the app does not read (${name})`);
}

function build(obj, state) {
  if (obj instanceof DType) {
    if (Array.isArray(state) && typeof state[1] === 'string') obj.order = state[1];
    return obj;
  }
  if (obj instanceof ArrayStub) {
    // (version, shape, dtype, is_fortran, raw data)
    if (!Array.isArray(state) || state.length < 5 || !(state[2] instanceof DType)) throw new PickleError('Unexpected NumPy array state');
    obj.array = decodeArray(state[1], state[2], state[4]);
    return obj;
  }
  throw new PickleError('This file contains a Python object the app does not read');
}

/** Replace every array stub in a finished result by the array it was filled with. */
function resolve(v, seen) {
  if (v instanceof ArrayStub) {
    if (!v.array) throw new PickleError('A NumPy array in this file was never filled in');
    return v.array;
  }
  if (Array.isArray(v)) {
    if (seen.has(v)) return v;
    seen.add(v);
    for (let i = 0; i < v.length; i++) v[i] = resolve(v[i], seen);
  } else if (v instanceof Map) {
    if (seen.has(v)) return v;
    seen.add(v);
    for (const [k, x] of v) v.set(k, resolve(x, seen));
  }
  return v;
}

/**
 * @param {Uint8Array|ArrayBuffer} input the pickle's bytes
 * @returns the top-level object
 */
export function unpickle(input) {
  const b = input instanceof Uint8Array ? input : new Uint8Array(input);
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let p = 0;
  const stack = [];
  const memo = [];
  const need = (n) => { if (p + n > b.length) throw new PickleError('The file ends in the middle of its data'); };
  const u8 = () => { need(1); return b[p++]; };
  const u16 = () => { need(2); const v = view.getUint16(p, true); p += 2; return v; };
  const u32 = () => { need(4); const v = view.getUint32(p, true); p += 4; return v; };
  const u64 = () => { need(8); const v = Number(view.getBigUint64(p, true)); p += 8; return v; };
  const take = (n) => { need(n); const out = b.subarray(p, p + n); p += n; return out; };
  const line = () => { const s = p; while (p < b.length && b[p] !== 10) p++; need(1); return latin1(b.subarray(s, p++)); };
  const popMark = () => {
    const at = stack.lastIndexOf(MARK);
    if (at < 0) throw new PickleError('Malformed pickle (no mark)');
    const items = stack.splice(at + 1);
    stack.pop();
    return items;
  };
  const top = () => { if (!stack.length) throw new PickleError('Malformed pickle (empty stack)'); return stack[stack.length - 1]; };
  const long = (n) => {
    const bytes = take(n);
    let v = 0;
    for (let i = n - 1; i >= 0; i--) v = v * 256 + bytes[i];
    return n && bytes[n - 1] & 0x80 ? v - Math.pow(256, n) : v;
  };

  for (;;) {
    const op = u8();
    switch (op) {
      case 0x80: u8(); break;                                   // PROTO
      case 0x95: take(8); break;                                // FRAME
      case 0x2e: return resolve(stack.pop(), new Set());         // STOP
      case 0x28: stack.push(MARK); break;                       // MARK
      case 0x4e: stack.push(null); break;                       // NONE
      case 0x88: stack.push(true); break;
      case 0x89: stack.push(false); break;
      case 0x4b: stack.push(u8()); break;                       // BININT1
      case 0x4d: stack.push(u16()); break;                      // BININT2
      case 0x4a: { need(4); stack.push(view.getInt32(p, true)); p += 4; break; }   // BININT
      case 0x8a: stack.push(long(u8())); break;                 // LONG1
      case 0x8b: stack.push(long(u32())); break;                // LONG4
      case 0x47: { need(8); stack.push(view.getFloat64(p, false)); p += 8; break; } // BINFLOAT
      case 0x8c: stack.push(utf8.decode(take(u8()))); break;    // SHORT_BINUNICODE
      case 0x58: stack.push(utf8.decode(take(u32()))); break;   // BINUNICODE
      case 0x8d: stack.push(utf8.decode(take(u64()))); break;   // BINUNICODE8
      case 0x43: stack.push(take(u8())); break;                 // SHORT_BINBYTES
      case 0x42: stack.push(take(u32())); break;                // BINBYTES
      case 0x8e: stack.push(take(u64())); break;                // BINBYTES8
      case 0x55: stack.push(take(u8())); break;                 // SHORT_BINSTRING (protocol <= 2 bytes)
      case 0x54: stack.push(take(u32())); break;                // BINSTRING
      case 0x5d: stack.push([]); break;                         // EMPTY_LIST
      case 0x7d: stack.push(new Map()); break;                  // EMPTY_DICT
      case 0x29: stack.push([]); break;                         // EMPTY_TUPLE
      case 0x85: stack.push([stack.pop()]); break;              // TUPLE1
      case 0x86: { const c = stack.pop(), a = stack.pop(); stack.push([a, c]); break; }
      case 0x87: { const c = stack.pop(), m = stack.pop(), a = stack.pop(); stack.push([a, m, c]); break; }
      case 0x74: stack.push(popMark()); break;                  // TUPLE
      case 0x6c: stack.push(popMark()); break;                  // LIST
      case 0x64: {                                              // DICT
        const items = popMark(), d = new Map();
        for (let i = 0; i + 1 < items.length; i += 2) d.set(items[i], items[i + 1]);
        stack.push(d); break;
      }
      case 0x61: { const v = stack.pop(); top().push(v); break; }            // APPEND
      case 0x65: { const items = popMark(); top().push(...items); break; }   // APPENDS
      case 0x73: { const v = stack.pop(), k = stack.pop(); top().set(k, v); break; }   // SETITEM
      case 0x75: {                                              // SETITEMS
        const items = popMark(), d = top();
        if (!(d instanceof Map)) throw new PickleError('Malformed pickle (items without a dict)');
        for (let i = 0; i + 1 < items.length; i += 2) d.set(items[i], items[i + 1]);
        break;
      }
      case 0x94: memo.push(top()); break;                       // MEMOIZE
      case 0x71: memo[u8()] = top(); break;                     // BINPUT
      case 0x72: memo[u32()] = top(); break;                    // LONG_BINPUT
      case 0x68: stack.push(memo[u8()]); break;                 // BINGET
      case 0x6a: stack.push(memo[u32()]); break;                // LONG_BINGET
      case 0x63: { const module = line(), name = line(); stack.push(new Global(module, name)); break; }   // GLOBAL
      case 0x93: { const name = stack.pop(), module = stack.pop(); stack.push(new Global(String(module), String(name))); break; }
      case 0x52: { const args = stack.pop(), fn = stack.pop(); stack.push(reduce(fn, args)); break; }      // REDUCE
      case 0x62: { const state = stack.pop(), obj = stack.pop(); stack.push(build(obj, state)); break; }   // BUILD
      case 0x30: stack.pop(); break;                            // POP
      case 0x32: stack.push(top()); break;                      // DUP
      default:
        throw new PickleError(p <= 2 ? 'Not a Python pickle file' : `This pickle uses a feature the app does not read (code 0x${op.toString(16)})`);
    }
  }
}
