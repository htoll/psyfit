// The image canvas: draws the composed scene plus vector overlays (fits, tracks,
// scale bar), with the colour bar in its own strip beside the image, and turns
// touches into pan / pinch-zoom / tap-select / double-tap-fit / press-and-hold
// pixel read-out. A tap on the colour bar opens the editor for its min and max.

import { composeScene, hueCss, makeNorm } from '../core/render.js';
import { applyAffine } from '../core/geometry.js';
import { SCOPES, channelRect, channelInfo } from '../core/layouts.js';
import { trackColor } from '../core/tracking.js';
import {
  S, subscribe, bump, buildScene, sceneFits, probeAt, select, frameKey, classify, CLASS_COLORS, summarize, brightnessSets,
  setScaleEditor, findFit, spectrumFor,
} from '../store.js';
import { THEME, fmtBrightness } from './theme.js';
import { niceTicks, fmtAxis } from './plots.js';

const TAP_SLOP = 9, DOUBLE_MS = 330, DOUBLE_DIST = 30, HOLD_MS = 420;
/** Width of the colour bar's strip, not counting the page gutter: ramp, tick marks and labels. */
const CBAR_STRIP = 50;

export class Viewer {
  constructor(host) {
    this.host = host;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'viewer-canvas';
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', 'Microscope image. Drag to pan, pinch to zoom, double-tap a spot to fit it.');
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.buf = document.createElement('canvas');
    this.bufCtx = this.buf.getContext('2d');
    this.image = null;
    this.view = { z: 1, tx: 0, ty: 0 };
    this.scene = null;
    this.sceneId = '';
    this.fits = [];
    this.cssW = 1; this.cssH = 1; this.dpr = 1;
    this.pointers = new Map();
    this.g = null;
    this.lastTap = null;
    this.holdTimer = null;
    this.raf = 0;
    this.needCompose = true;
    this.edge = 12;               // clearance from the right edge (more beside a notch)
    this.area = { x: 0, y: 0, w: 1, h: 1 };   // the part of the stage the image may occupy
    this.strip = 0;               // width kept free at the right for the colour bar
    this.inset = 0;               // height of the card open at the bottom of the stage
    this.insetWanted = 0;
    this.insetTimer = null;
    this.insetAt = 0;             // when the card last changed
    this.movedAt = 0;             // when the user last panned or zoomed
    this.lift = 0;                // how far the image has been moved up to clear the card
    this.glide = 0;               // animation frame of that move
    this.cbar = null;             // where the colour bar was last drawn
    this.ramps = new Map();       // LUT -> 1 x 256 canvas

    this.onDown = this.onDown.bind(this); this.onMove = this.onMove.bind(this); this.onUp = this.onUp.bind(this);
    const c = this.canvas;
    c.addEventListener('pointerdown', this.onDown);
    c.addEventListener('pointermove', this.onMove);
    c.addEventListener('pointerup', this.onUp);
    c.addEventListener('pointercancel', this.onUp);
    c.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('lostpointercapture', this.onUp);
    // The fingers actually on the image are the truth. A touch can end without the page being told
    // (the system or the app around the page takes the gesture over); left in our books it would turn
    // every later one-finger drag into a pinch.
    const sync = (e) => this.syncTouches(e.targetTouches ? e.targetTouches.length : 0);
    c.addEventListener('touchstart', sync, { passive: true });
    c.addEventListener('touchcancel', sync, { passive: true });
    // A tap here can open a card under the finger. Without preventDefault the browser would follow the
    // touch with a click at the same spot, which would land on that card (and could press its Close).
    c.addEventListener('touchend', (e) => { if (e.cancelable) e.preventDefault(); sync(e); }, { passive: false });
    this.onLeave = () => { if (document.hidden !== false || !document.hasFocus()) this.cancelTouches(); };
    document.addEventListener('visibilitychange', this.onLeave);
    window.addEventListener('pagehide', this.onLeave);
    window.addEventListener('blur', this.onLeave);
    // Stop Safari's own page pinch-zoom from competing with ours.
    for (const t of ['gesturestart', 'gesturechange', 'gestureend']) c.addEventListener(t, (e) => e.preventDefault());

    this.ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.resize()) : null;
    if (this.ro) this.ro.observe(host); else window.addEventListener('resize', () => this.resize());
    this.unsub = subscribe((what) => {
      if (what.scene) this.needCompose = true;
      if (what.scene || what.overlay) this.schedule();
    });
    this.resize();
  }

  destroy() {
    if (this.ro) this.ro.disconnect();
    this.unsub();
    clearTimeout(this.insetTimer);
    clearTimeout(this.holdTimer);
    cancelAnimationFrame(this.raf);
    cancelAnimationFrame(this.glide);
    document.removeEventListener('visibilitychange', this.onLeave);
    window.removeEventListener('pagehide', this.onLeave);
    window.removeEventListener('blur', this.onLeave);
    this.canvas.remove();
  }

  // ---- geometry -----------------------------------------------------------

  resize() {
    const r = this.host.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    this.edge = Math.max(12, sideGutter(this.host) - 4);
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
    this.cssW = w; this.cssH = h; this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
    this.relayout();
    this.schedule();
  }

  /**
   * How the stage divides: the area the image lives in and, when the colour
   * bar is on, a strip to its right that belongs to the bar alone. The image
   * is fitted, panned and clipped inside the area, so it never runs under the bar.
   */
  layoutFor(scene) {
    const full = { x: 0, y: 0, w: this.cssW, h: this.cssH };
    if (!scene || !S.display.showColorbar) return { area: full, strip: 0 };
    const strip = this.edge + CBAR_STRIP;
    if (this.cssW - strip < 120 || this.cssH < 80) return { area: full, strip: 0 };
    return { area: { x: 0, y: 0, w: this.cssW - strip, h: this.cssH }, strip };
  }

  /**
   * Tell the viewer how much of the stage's bottom a card (fit details,
   * colour scale) covers. The image then slides up clear of the card as far as
   * there is room above it, and back when the card goes. Its size never
   * changes: only two fingers, the fit button or a new layout zoom. The move
   * waits a moment so that the second tap of a double-tap still lands on the
   * spot it was aimed at.
   */
  setBottomInset(px, selection = '') {
    const want = Math.max(0, Math.round(px));
    if (want === this.insetWanted && selection === this.insetFor) return;
    const sameCard = want === this.insetWanted;                 // only what the card is about changed
    this.insetWanted = want; this.insetFor = selection;
    if (sameCard && this.movedAt > this.insetAt) return;        // the user has placed the image: leave it there
    this.insetAt = performance.now();
    clearTimeout(this.insetTimer);
    this.insetTimer = setTimeout(() => { this.inset = this.insetWanted; this.settleForCard(true); }, want > this.inset || sameCard ? 380 : 140);
  }

  /**
   * Vertical extent on screen, with the image at `rest`, of what the open card
   * is about: the selected fits and, for a Nikon particle, the strip its
   * spectrum is read from. Null when nothing is selected.
   */
  selectionSpan(rest) {
    if (!S.sel || !S.sel.ids || !this.scene) return null;
    const dy = rest - this.view.ty;
    let top = Infinity, bottom = -Infinity;
    const see = (y) => { if (y + dy < top) top = y + dy; if (y + dy > bottom) bottom = y + dy; };
    for (const item of this.fits) {
      if (!S.sel.ids.includes(item.fit.id)) continue;
      see(this.toScreen(item.X, item.Y)[1]);
      const L = this.scene.layers[item.layerIndex];
      const sp = L && !L.inv ? spectrumFor(item.fit) : null;
      if (sp && sp.status === 'ok' && sp.win.y1 >= L.rect.y && sp.win.y0 <= L.rect.y + L.rect.h - 1) {
        see(this.toScreen(0, sp.win.y1 - L.rect.y)[1]); see(this.toScreen(0, sp.win.y0 - L.rect.y)[1]);
      }
    }
    return bottom >= top ? [top, bottom] : null;
  }

  /**
   * Slide the image clear of the open card, or back. Skipped while a finger is
   * on the image (it runs when the finger lifts) and once the user has moved
   * the image themselves since the card changed: the view is theirs then.
   */
  settleForCard(animate) {
    if (!this.scene) return;
    if (this.fingers().length) { this.settleLater = true; return; }
    this.settleLater = false;
    if (this.movedAt > this.insetAt) return;
    const a = this.area, v = this.view;
    const rest = v.ty + this.lift;                              // where the image sits without the lift
    let lift = 0;
    const free = a.y + a.h - this.inset;                        // bottom of what the card leaves visible
    if (this.inset && free - a.y >= 100) {
      // as far as the image has room above it ...
      lift = Math.min(Math.max(0, rest + this.scene.h * v.z - free), Math.max(0, rest - a.y));
      // ... and further if what the card is about would still be under it, but never so far that it leaves the top
      const span = this.selectionSpan(rest);
      if (span) lift = Math.max(lift, Math.min(span[1] + 22 - free, span[0] - 22 - a.y));
      lift = Math.max(0, lift);
    }
    if (Math.abs(lift - this.lift) < 0.5) return;
    this.lift = lift;
    this.glideTo(rest - lift, animate);
  }

  /** Move the image vertically to `ty`, over a fifth of a second unless told not to. */
  glideTo(ty, animate) {
    cancelAnimationFrame(this.glide);
    const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!animate || reduce) { this.view.ty = ty; this.schedule(); return; }
    const from = this.view.ty, t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 200), e = 1 - (1 - k) * (1 - k);
      this.view.ty = from + (ty - from) * e;
      this.draw();
      this.glide = k < 1 ? requestAnimationFrame(step) : 0;
    };
    this.glideEnd = ty;
    this.glide = requestAnimationFrame(step);
  }

  /** Finish a glide at once (a finger has come down, or the view is being set). */
  endGlide() {
    if (!this.glide) return;
    cancelAnimationFrame(this.glide);
    this.glide = 0;
    this.view.ty = this.glideEnd;
  }

  /** The user has panned or zoomed: forget the card's lift, so nothing is "put back" later. */
  userMoved() {
    this.movedAt = performance.now();
    this.lift = 0;
  }

  /**
   * Apply the current division of the stage. A view that showed the whole
   * image still shows it afterwards; a zoomed view keeps its centre.
   */
  relayout() {
    const next = this.layoutFor(this.scene);
    const old = this.area;
    if (old.w === next.area.w && old.h === next.area.h && this.strip === next.strip) return;
    const keep = this.scene && !this.isFitted() ? this.toScene(old.x + old.w / 2, old.y + old.h / 2) : null;
    this.area = next.area; this.strip = next.strip;
    // Notices and cards over the stage keep out of the strip too (see --cbar in app.css).
    if (this.host.parentElement) this.host.parentElement.style.setProperty('--cbar', `${this.strip}px`);
    if (!this.scene) return;
    if (keep) {
      this.endGlide();
      const [sx, sy] = this.toScreen(keep[0], keep[1]);
      this.view.tx += this.area.x + this.area.w / 2 - sx; this.view.ty += this.area.y + this.area.h / 2 - sy;
      this.clampView();
    } else this.applyFit();
  }

  fitZoom() {
    if (!this.scene) return 1;
    return Math.min(this.area.w / this.scene.w, this.area.h / this.scene.h) * 0.985;
  }

  applyFit() {
    if (!this.scene) return;
    this.endGlide();
    const a = this.area, z = this.fitZoom();
    this.view = { z, tx: a.x + (a.w - this.scene.w * z) / 2, ty: a.y + (a.h - this.scene.h * z) / 2 };
    // A fresh fit is not the user's own placement: let an open card lift it again.
    this.lift = 0; this.movedAt = 0;
    if (this.inset) this.settleForCard(false);
  }

  fitView() {
    this.applyFit();
    this.schedule();
  }

  isFitted() {
    return this.scene ? Math.abs(this.view.z - this.fitZoom()) < 1e-6 : true;
  }

  toScreen(X, Y) {
    const v = this.view;
    return [v.tx + (X + 0.5) * v.z, v.ty + (this.scene.h - 0.5 - Y) * v.z];
  }

  toScene(sx, sy) {
    const v = this.view;
    return [(sx - v.tx) / v.z - 0.5, this.scene.h - 0.5 - (sy - v.ty) / v.z];
  }

  clampView() {
    if (!this.scene) return;
    const v = this.view, a = this.area, fit = this.fitZoom();
    v.z = Math.min(Math.max(v.z, fit * 0.5), Math.max(64, fit * 4));
    const w = this.scene.w * v.z, h = this.scene.h * v.z, m = 48;
    v.tx = Math.min(a.x + a.w - m, Math.max(a.x + m - w, v.tx));
    v.ty = Math.min(a.y + a.h - m, Math.max(a.y + m - h, v.ty));
  }

  zoomAt(sx, sy, factor) {
    if (!this.scene) return;
    this.endGlide();
    this.userMoved();
    const v = this.view;
    const ix = (sx - v.tx) / v.z, iy = (sy - v.ty) / v.z;
    v.z *= factor;
    this.clampView();
    v.tx = sx - ix * v.z; v.ty = sy - iy * v.z;
    this.clampView();
    this.schedule();
  }

  zoomBy(factor) { this.zoomAt(this.area.x + this.area.w / 2, this.area.y + this.area.h / 2, factor); }

  // ---- drawing --------------------------------------------------------------

  schedule() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
  }

  compose() {
    this.needCompose = false;
    const scene = buildScene();
    const prevId = this.sceneId;
    this.scene = scene;
    if (!scene) { this.sceneId = ''; return; }
    this.sceneId = `${scene.w}x${scene.h}:${scene.baseFile.id}:${scene.base.channel}`;
    if (this.buf.width !== scene.w || this.buf.height !== scene.h || !this.image) {
      this.buf.width = scene.w; this.buf.height = scene.h;
      this.image = this.bufCtx.createImageData(scene.w, scene.h);
    }
    composeScene(scene, this.image.data);
    this.bufCtx.putImageData(this.image, 0, 0);
    if (prevId !== this.sceneId) this.applyFit();
  }

  draw() {
    if (this.needCompose) this.compose();
    this.relayout();          // the colour bar's strip comes and goes with the scene and its setting
    const ctx = this.ctx, dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = THEME.canvas;
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    const scene = this.scene;
    this.cbar = null;
    if (!scene) return;
    const v = this.view, a = this.area;
    ctx.save();
    ctx.beginPath(); ctx.rect(a.x, a.y, a.w, a.h); ctx.clip();
    ctx.imageSmoothingEnabled = v.z * dpr < 1;
    ctx.drawImage(this.buf, v.tx, v.ty, scene.w * v.z, scene.h * v.z);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (scene.sensor && S.display.showGrid) this.drawChannelGrid(ctx, scene);
    if (S.movie.showTracks) this.drawTracks(ctx, scene);
    this.fits = S.display.showFits ? sceneFits(scene) : [];
    this.drawFits(ctx, scene);
    this.drawSpectralWindows(ctx, scene);
    if (S.display.showScaleBar) this.drawScaleBar(ctx, scene);
    ctx.restore();
    if (this.strip) {
      this.cbar = this.colorbarBox(scene);
      if (this.cbar) this.drawColorbar(ctx, scene, this.cbar);
    }
    if (S.inspect) this.drawInspect(ctx);
  }

  // ---- colour bar -------------------------------------------------------------

  /**
   * The colour bar inside its strip, in CSS pixels: one vertical ramp per
   * layer, as tall as the part of the image in view (like the colour bar of a
   * figure). Null when the strip is too short for one.
   */
  colorbarBox(scene) {
    const a = this.area, v = this.view, n = scene.layers.length;
    const thick = n > 1 ? 7 : 10, gap = 3;
    const span = n * thick + (n - 1) * gap;
    const top = a.y + 26, bottom = a.y + a.h - 10;           // room above for the unit
    let y = Math.max(top, v.ty), end = Math.min(bottom, v.ty + scene.h * v.z);
    if (end - y < 72) {                                      // little of the image in view: centre a short bar
      const len = Math.min(200, bottom - top);
      y = top + (bottom - top - len) / 2; end = y + len;
    }
    if (end - y < 36) return null;
    return { x: this.cssW - this.edge - span, y, len: end - y, thick, gap, span, n };
  }

  /** True when a touch at (sx, sy) is in the colour bar's strip. */
  inColorbar(sx) {
    return this.strip > 0 && sx >= this.cssW - this.strip;
  }

  /** A LUT as a 1 x 256 strip, brightest at the top. */
  ramp(lut) {
    let c = this.ramps.get(lut);
    if (!c) {
      c = document.createElement('canvas');
      c.width = 1; c.height = 256;
      const g = c.getContext('2d'), img = g.createImageData(1, 256);
      for (let i = 0; i < 256; i++) {
        const k = (255 - i) * 3;
        img.data[i * 4] = lut[k]; img.data[i * 4 + 1] = lut[k + 1]; img.data[i * 4 + 2] = lut[k + 2]; img.data[i * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      this.ramps.set(lut, c);
    }
    return c;
  }

  /**
   * Colour scale in pps. One image: a ramp labelled with its min, its max and
   * round values between them. An overlay: one ramp per layer in the layer's
   * colour, without numbers (the editor that a tap opens lists each range).
   */
  drawColorbar(ctx, scene, b) {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    scene.layers.forEach((L, i) => {
      const x = b.x + i * (b.thick + b.gap);
      ctx.drawImage(this.ramp(L.lut), x, b.y, b.thick, b.len);
      ctx.lineWidth = 1;
      ctx.strokeStyle = S.scale ? THEME.accent : 'rgba(255,255,255,0.6)';
      ctx.strokeRect(x - 0.5, b.y - 0.5, b.thick + 1, b.len + 1);
    });
    ctx.fillStyle = THEME.ink;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.font = `600 10.5px ${THEME.mono}`;
    ctx.fillText(S.display.log ? 'pps, log' : 'pps', b.x + b.span + 1, b.y - 13);
    if (scene.layers.length === 1) {
      const L = scene.layers[0];
      const norm = makeNorm(L.lo, L.hi, L.log);
      const marks = [{ v: L.hi, y: b.y, ty: b.y + 5 }, { v: L.lo, y: b.y + b.len, ty: b.y + b.len - 5 }];
      for (const val of scaleTicks(L.lo, L.hi, L.log, b.len)) {
        const y = b.y + (1 - norm.fwd(val)) * b.len;
        if (marks.every((m) => Math.abs(m.ty - y) >= 15)) marks.push({ v: val, y, ty: y });
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1;
      for (const m of marks) {
        const ly = Math.min(b.y + b.len - 0.5, Math.max(b.y + 0.5, Math.round(m.y) + 0.5));
        ctx.beginPath(); ctx.moveTo(b.x - 4, ly); ctx.lineTo(b.x, ly); ctx.stroke();
        ctx.fillText(fmtAxis(m.v), b.x - 7, m.ty);
      }
    }
    ctx.restore();
  }

  drawChannelGrid(ctx, scene) {
    const file = scene.baseFile;
    const scope = SCOPES[file.scope];
    if (!scope || !scope.channels.length) return;
    ctx.save();
    ctx.font = `600 11px ${THEME.sans}`;
    ctx.textBaseline = 'top';
    for (const ch of scope.channels) {
      const r = channelRect(file.scope, ch.id, file.src);
      if (!r) continue;
      const [x0, y1] = this.toScreen(r.x - 0.5, r.y - 0.5);
      const [x1, y0] = this.toScreen(r.x + r.w - 0.5, r.y + r.h - 0.5);
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.38)';
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
      ctx.setLineDash([]);
      const label = ch.label, tw = ctx.measureText(label).width;
      const lx = Math.max(x0 + 5, 5), ly = Math.max(y0 + 5, 5);
      if (lx + tw + 22 < x1 && ly + 18 < y1) {
        ctx.fillStyle = 'rgba(11,14,20,0.72)';
        roundRect(ctx, lx, ly, tw + 21, 18, 9); ctx.fill();
        ctx.fillStyle = ch.color;
        ctx.beginPath(); ctx.arc(lx + 9, ly + 9, 3.5, 0, 6.2832); ctx.fill();
        ctx.fillStyle = THEME.ink;
        ctx.fillText(label, lx + 16, ly + 3.5);
      }
    }
    ctx.restore();
  }

  drawFits(ctx, scene) {
    const list = this.fits;
    if (!list.length) return;
    const z = this.view.z;
    const selected = new Set(S.sel && S.sel.ids ? S.sel.ids : []);
    const labels = S.display.showLabels && (list.length <= 70 || z >= 3.5);
    let single = null;
    if (S.display.colorBy === 'class') {
      const sets = brightnessSets('file');
      single = {};
      for (const s of sets) { const sm = summarize(s.values, 2, 'lowest'); if (sm) single[s.key] = sm.mu; }
    }
    ctx.save();
    ctx.font = `500 10.5px ${THEME.mono}`;
    ctx.textBaseline = 'middle';
    const placed = [];
    const free = (x, y, w) => {
      for (const b of placed) if (x < b[0] + b[2] + 2 && x + w + 2 > b[0] && Math.abs(y - b[1]) < 12) return false;
      placed.push([x, y, w]);
      return true;
    };
    // selected fits are labelled first so they always keep their label
    const ordered = selected.size ? list.filter((i) => selected.has(i.fit.id)).concat(list.filter((i) => !selected.has(i.fit.id))) : list;
    for (const item of ordered) {
      const { fit, X, Y } = item;
      const [sx, sy] = this.toScreen(X, Y);
      if (sx < -40 || sy < -40 || sx > this.cssW + 40 || sy > this.cssH + 40) continue;
      const r = Math.max(2 * Math.max(fit.sx, fit.sy) * z, 5);
      const isSel = selected.has(fit.id);
      let color = 'rgba(255,255,255,0.82)';
      if (scene.multi) color = hueCss(scene.layers[item.layerIndex].layer.hue);
      if (single && single[fit.channel]) color = CLASS_COLORS[classify([fit.brightness], single[fit.channel]).indexOf(1)] || color;
      if (fit.coloc) color = '#7dff5a';
      if (fit.manual) color = THEME.accent;
      if (!fit.ok) color = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = isSel ? 2.5 : 1.25;
      ctx.strokeStyle = isSel ? THEME.accent : color;
      if (!fit.ok) ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.2832); ctx.stroke();
      ctx.setLineDash([]);
      if (isSel) {
        ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(242,184,75,0.45)';
        ctx.beginPath(); ctx.arc(sx, sy, r + 4, 0, 6.2832); ctx.stroke();
      }
      if (labels || isSel) {
        const text = fmtBrightness(fit.brightness, true);
        const tx = sx + r + 4, ty = sy - r * 0.55;
        if (!free(tx, ty, ctx.measureText(text).width) && !isSel) continue;
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(8,10,15,0.8)';
        ctx.strokeText(text, tx, ty);
        ctx.fillStyle = isSel ? THEME.accent : THEME.ink;
        ctx.fillText(text, tx, ty);
      }
    }
    ctx.restore();
  }

  /**
   * For a selected particle of a Nikon frame: outline the strip its spectrum is
   * read from, join the two with a line, and outline (dashed) the strips of
   * other particles whose spectra may run into it.
   */
  drawSpectralWindows(ctx, scene) {
    if (!S.sel || !S.sel.ids) return;
    for (const id of S.sel.ids) {
      const fit = findFit(id);
      const sp = fit ? spectrumFor(fit) : null;
      if (!sp || sp.status !== 'ok') continue;
      const L = scene.layers.find((x) => x.file.id === fit.fileId && !x.inv);
      if (!L) continue;
      const strip = (win) => {
        const left = this.toScreen(win.x0 - L.rect.x, 0)[0], right = this.toScreen(win.x1 - L.rect.x, 0)[0];
        const top = this.toScreen(0, win.y1 - L.rect.y + 0.5)[1], bottom = this.toScreen(0, win.y0 - L.rect.y - 0.5)[1];
        return { left, right, top, bottom, inView: win.x1 >= L.rect.x && win.x0 <= L.rect.x + L.rect.w - 1 && win.y1 >= L.rect.y && win.y0 <= L.rect.y + L.rect.h - 1 };
      };
      ctx.save();
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = '#f0c36a';
      for (const o of sp.overlaps) {
        const r = strip(o.win);
        if (r.inView) ctx.strokeRect(r.left, r.top - 1, r.right - r.left, r.bottom - r.top + 2);
      }
      ctx.setLineDash([]);
      const r = strip(sp.win);
      if (r.inView) {
        ctx.strokeStyle = THEME.accent; ctx.lineWidth = 1.75;
        ctx.strokeRect(r.left, r.top - 1.5, r.right - r.left, r.bottom - r.top + 3);
        const px = fit.x - L.rect.x, py = fit.y - L.rect.y;
        if (px >= 0 && py >= 0 && px <= L.rect.w - 1 && py <= L.rect.h - 1) {
          const [fx, fy] = this.toScreen(px, py);
          ctx.strokeStyle = 'rgba(242,184,75,0.45)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(r.left, (r.top + r.bottom) / 2); ctx.stroke();
        }
      }
      ctx.restore();
    }
  }

  drawTracks(ctx, scene) {
    const tail = S.movie.tail;
    ctx.save();
    for (const L of scene.layers) {
      const file = L.file;
      if (!file.tracks || !file.tracks.list.length) continue;
      const cur = file.view === 'frame' ? file.cur : file.loaded.length - 1;
      const first = file.view === 'frame' && tail > 0 ? cur - tail : -Infinity;
      const selId = S.sel && S.sel.trackId && S.sel.fileId === file.id ? S.sel.trackId : null;
      for (const t of file.tracks.list) {
        if (L.layer.channel !== 'all' && t.channel !== L.layer.channel) continue;
        const pts = t.points;
        if (pts[0].f > cur || pts[pts.length - 1].f < first) continue;
        const color = S.movie.colorMode === 'channel' ? channelInfo(file.scope, t.channel).color : trackColor(t.id);
        const isSel = selId === t.id;
        ctx.strokeStyle = color;
        ctx.lineWidth = isSel ? 3 : 1.6;
        let prev = null, head = null;
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          if (p.f > cur) break;
          if (p.f < first && !isSel) continue;
          const lx = p.x - L.rect.x, ly = p.y - L.rect.y;
          const [X, Y] = L.inv ? applyAffine(L.m, lx, ly) : [lx, ly];
          const s = this.toScreen(X, Y);
          if (prev) {
            const age = tail > 0 && Number.isFinite(first) ? (cur - p.f) / Math.max(1, tail) : 0;
            ctx.globalAlpha = isSel ? 1 : Math.max(0.18, 1 - 0.8 * Math.min(1, age));
            ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(s[0], s[1]); ctx.stroke();
          }
          prev = s; head = p;
        }
        if (head && cur - head.f <= 2) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = color;
          ctx.beginPath(); ctx.arc(prev[0], prev[1], isSel ? 4.5 : 3, 0, 6.2832); ctx.fill();
          ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(8,10,15,0.85)';
          ctx.stroke();
          if (isSel) {
            ctx.font = `600 11px ${THEME.mono}`; ctx.textBaseline = 'middle';
            ctx.lineWidth = 3; ctx.strokeText(`#${t.id}`, prev[0] + 8, prev[1] - 8);
            ctx.fillStyle = THEME.ink; ctx.fillText(`#${t.id}`, prev[0] + 8, prev[1] - 8);
          }
        }
      }
    }
    ctx.restore();
  }

  drawScaleBar(ctx, scene) {
    const pix = scene.baseFile.pixUm;
    if (!(pix > 0)) return;
    const z = this.view.z;
    const a = this.area;
    const target = (a.w * 0.22) / z * pix;                          // micrometres
    const k = Math.pow(10, Math.floor(Math.log10(target)));
    const um = [5, 2, 1].map((m) => m * k).find((val) => val <= target) || k;
    const len = (um / pix) * z;
    const covered = this.inset && a.h - this.inset >= 100 ? this.inset : 0;      // keep clear of an open card
    const x1 = a.x + a.w - (this.strip ? 10 : this.edge + 2), y = a.y + a.h - covered - 16;
    ctx.save();
    ctx.lineCap = 'butt';
    ctx.strokeStyle = 'rgba(8,10,15,0.75)'; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x1 - len - 1, y); ctx.lineTo(x1 + 1, y); ctx.stroke();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(x1 - len, y); ctx.lineTo(x1, y); ctx.stroke();
    const text = um >= 1 ? `${um} µm` : `${Math.round(um * 1000)} nm`;
    ctx.font = `600 11px ${THEME.sans}`; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(8,10,15,0.75)';
    ctx.strokeText(text, x1, y - 5);
    ctx.fillStyle = '#ffffff'; ctx.fillText(text, x1, y - 5);
    ctx.restore();
  }

  drawInspect(ctx) {
    const { sx, sy, lines } = S.inspect;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx - 14, sy); ctx.lineTo(sx - 4, sy); ctx.moveTo(sx + 4, sy); ctx.lineTo(sx + 14, sy);
    ctx.moveTo(sx, sy - 14); ctx.lineTo(sx, sy - 4); ctx.moveTo(sx, sy + 4); ctx.lineTo(sx, sy + 14);
    ctx.stroke();
    ctx.font = `500 12px ${THEME.mono}`;
    const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
    const h = lines.length * 16 + 10;
    let bx = sx - w / 2, by = sy - 70 - h;
    if (by < 6) by = sy + 46;
    bx = Math.max(6, Math.min(this.cssW - w - 6, bx));
    ctx.fillStyle = 'rgba(11,14,20,0.9)';
    roundRect(ctx, bx, by, w, h, 8); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.16)'; ctx.stroke();
    ctx.fillStyle = THEME.ink; ctx.textBaseline = 'top';
    lines.forEach((l, i) => ctx.fillText(l, bx + 8, by + 6 + i * 16));
    ctx.restore();
  }

  /** PNG of what is on screen (image plus overlays). */
  toBlob() {
    this.draw();
    return new Promise((resolve) => this.canvas.toBlob(resolve, 'image/png'));
  }

  // ---- input --------------------------------------------------------------

  pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  /** Fingers that count for gestures: every pointer on the image except a resting grip. */
  fingers() {
    const out = [];
    for (const r of this.pointers.values()) if (!r.rest) out.push(r);
    return out;
  }

  /**
   * A finger that has sat still at the side of the stage for a while is a hand
   * holding the phone, not half of a pinch.
   */
  isResting(r, now) {
    if (r.type === 'mouse') return false;
    const atSide = r.x0 < 22 || r.x0 > this.cssW - 22 || (this.strip > 0 && r.x0 >= this.cssW - this.strip);
    return atSide && now - r.t0 > 600 && Math.hypot(r.x - r.x0, r.y - r.y0) <= TAP_SLOP;
  }

  /** Start the gesture of a single finger: a tap, a hold or a drag, decided as it goes. */
  beginSingle(r) {
    clearTimeout(this.holdTimer);
    // A touch that starts in the colour bar's strip opens its editor if it ends as a tap; a drag still pans.
    const onBar = this.inColorbar(r.x);
    this.g = { mode: 'pending', onBar, x0: r.x, y0: r.y, lx: r.x, ly: r.y, t0: performance.now() };
    if (!onBar) {
      this.holdTimer = setTimeout(() => {
        if (this.g && this.g.mode === 'pending') { this.g.mode = 'inspect'; this.inspectAt(this.g.lx, this.g.ly); }
      }, HOLD_MS);
    }
  }

  /** Forget every touch (the page lost focus, or was hidden, mid-gesture). */
  cancelTouches() {
    this.pointers.clear();
    clearTimeout(this.holdTimer);
    this.g = null;
    this.clearInspect();
  }

  /**
   * Reconcile our pointers with the number of fingers the browser reports on
   * the image. Any touch we still hold beyond that count ended without a
   * pointerup or pointercancel; the oldest ones are the ones that are gone.
   */
  syncTouches(count) {
    const held = [];
    for (const [id, r] of this.pointers) if (r.type !== 'mouse') held.push(id);
    if (held.length <= count) return;
    for (const id of held.slice(0, held.length - count)) this.pointers.delete(id);
    this.regroup();
  }

  /** Put the gesture in line with the fingers that remain after some were dropped. */
  regroup() {
    const f = this.fingers();
    if (!f.length) {
      clearTimeout(this.holdTimer);
      this.g = null;
      this.clearInspect();
      this.afterTouch();
    } else if (f.length === 1 && (!this.g || this.g.mode === 'pinch')) {
      const r = f[0];
      // just landed: a fresh single-finger gesture; otherwise it carries on as a drag
      if (performance.now() - r.t0 < 250) this.beginSingle(r);
      else this.g = { mode: 'pan', x0: r.x, y0: r.y, lx: r.x, ly: r.y, t0: performance.now() };
    }
  }

  /** Every finger is off the image: run what was waiting for that. */
  afterTouch() {
    if (this.settleLater) this.settleForCard(true);
  }

  onDown(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try { this.canvas.setPointerCapture(e.pointerId); } catch (err) { /* older WebKit */ }
    // Touching the image finishes any number being typed (and puts the keyboard away).
    const active = document.activeElement;
    if (active && active !== document.body && typeof active.blur === 'function') active.blur();
    this.endGlide();
    const now = performance.now();
    // A primary pointer is the first of its kind in contact: any other of that kind still in our
    // books ended without telling us.
    if (e.isPrimary) for (const [id, r] of this.pointers) if (r.type === e.pointerType) this.pointers.delete(id);
    // A grip resting at the side does not take part in what the new finger does.
    let rested = false;
    for (const r of this.pointers.values()) if (!r.rest && this.isResting(r, now)) { r.rest = true; rested = true; }
    if (rested) { clearTimeout(this.holdTimer); this.g = null; this.clearInspect(); }
    const p = this.pos(e);
    const rec = { x: p.x, y: p.y, x0: p.x, y0: p.y, t0: now, type: e.pointerType, rest: false };
    this.pointers.set(e.pointerId, rec);
    const n = this.fingers().length;
    if (n === 1) this.beginSingle(rec);
    else if (n === 2) {
      clearTimeout(this.holdTimer);
      this.clearInspect();
      this.startPinch();
    }
    e.preventDefault();
  }

  startPinch() {
    const [a, b] = this.fingers();
    this.g = {
      mode: 'pinch', d0: Math.max(10, Math.hypot(a.x - b.x, a.y - b.y)),
      mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, view0: { ...this.view },
    };
  }

  onMove(e) {
    const rec = this.pointers.get(e.pointerId);
    if (!rec) return;
    const p = this.pos(e);
    rec.x = p.x; rec.y = p.y;
    const g = this.g;
    if (rec.rest || !g || !this.scene) return;
    const f = this.fingers();
    if (g.mode === 'pinch') {
      if (f.length < 2) return;
      const [a, b] = f;
      const d = Math.max(10, Math.hypot(a.x - b.x, a.y - b.y));
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const v0 = g.view0;
      const ix = (g.mx - v0.tx) / v0.z, iy = (g.my - v0.ty) / v0.z;
      this.userMoved();
      this.view.z = v0.z * (d / g.d0);
      this.clampView();
      this.view.tx = mx - ix * this.view.z; this.view.ty = my - iy * this.view.z;
      this.clampView();
      this.schedule();
      return;
    }
    if (g.mode === 'pending' && Math.hypot(p.x - g.x0, p.y - g.y0) > TAP_SLOP) {
      g.mode = 'pan';
      clearTimeout(this.holdTimer);
    }
    if (g.mode === 'pan') {
      this.userMoved();
      this.view.tx += p.x - g.lx; this.view.ty += p.y - g.ly;
      this.clampView();
      this.schedule();
    } else if (g.mode === 'inspect') {
      this.inspectAt(p.x, p.y);
    }
    g.lx = p.x; g.ly = p.y;
  }

  onUp(e) {
    const rec = this.pointers.get(e.pointerId);
    if (!rec) return;
    this.pointers.delete(e.pointerId);
    if (rec.rest) { if (!this.pointers.size) this.afterTouch(); return; }
    clearTimeout(this.holdTimer);
    const g = this.g;
    const f = this.fingers();
    if (!g) { if (!f.length) this.afterTouch(); return; }
    if (g.mode === 'pinch') {
      if (f.length === 1) {
        const q = f[0];
        this.g = { mode: 'pan', x0: q.x, y0: q.y, lx: q.x, ly: q.y, t0: performance.now() };
      } else if (!f.length) { this.g = null; this.afterTouch(); }
      return;
    }
    this.g = null;
    if (g.mode === 'inspect') this.clearInspect();
    else if (g.mode === 'pending' && e.type === 'pointerup' && this.scene) {
      if (g.onBar) { this.lastTap = null; setScaleEditor(!S.scale); } else this.onTap(rec.x, rec.y);
    }
    if (!this.fingers().length) this.afterTouch();
  }

  onWheel(e) {
    e.preventDefault();
    const p = this.pos(e);
    const k = e.ctrlKey ? 0.012 : 0.0018;
    this.zoomAt(p.x, p.y, Math.exp(-e.deltaY * k));
  }

  onTap(sx, sy) {
    const now = performance.now();
    const last = this.lastTap;
    if (last && now - last.t < DOUBLE_MS && Math.hypot(sx - last.x, sy - last.y) < DOUBLE_DIST) {
      this.lastTap = null;
      const [X, Y] = this.toScene(sx, sy);
      if (X < -0.5 || Y < -0.5 || X > this.scene.w - 0.5 || Y > this.scene.h - 0.5) return;
      // A fingertip covers several image pixels when zoomed out: look that far for the peak.
      const snap = Math.min(12, Math.max(2, 16 / this.view.z));
      probeAt(X, Y, snap);
      return;
    }
    this.lastTap = { t: now, x: sx, y: sy };
    // Single tap: select the nearest marker, or clear the selection.
    let best = null, bd = 24;
    for (const item of this.fits) {
      const [fx, fy] = this.toScreen(item.X, item.Y);
      const d = Math.hypot(fx - sx, fy - sy);
      if (d < bd) { bd = d; best = item; }
    }
    if (best) {
      const group = best.fit.group;
      const ids = group ? this.fits.filter((i) => i.fit.group === group).map((i) => i.fit.id) : [best.fit.id];
      select({ ids });
      return;
    }
    const track = S.movie.showTracks ? this.trackNear(sx, sy) : null;
    if (track) { select({ trackId: track.id, fileId: track.fileId }); return; }
    if (S.sel) select(null);
  }

  trackNear(sx, sy) {
    const scene = this.scene;
    let best = null, bd = 20;
    for (const L of scene.layers) {
      const file = L.file;
      if (!file.tracks) continue;
      const cur = file.view === 'frame' ? file.cur : file.loaded.length - 1;
      const first = file.view === 'frame' && S.movie.tail > 0 ? cur - S.movie.tail : -Infinity;
      for (const t of file.tracks.list) {
        if (L.layer.channel !== 'all' && t.channel !== L.layer.channel) continue;
        for (const p of t.points) {
          if (p.f > cur || p.f < first) continue;
          const lx = p.x - L.rect.x, ly = p.y - L.rect.y;
          const [X, Y] = L.inv ? applyAffine(L.m, lx, ly) : [lx, ly];
          const s = this.toScreen(X, Y);
          const d = Math.hypot(s[0] - sx, s[1] - sy);
          if (d < bd) { bd = d; best = { id: t.id, fileId: file.id }; }
        }
      }
    }
    return best;
  }

  inspectAt(sx, sy) {
    const scene = this.scene;
    if (!scene) return;
    const [X, Y] = this.toScene(sx, sy);
    const lines = [];
    for (const L of scene.layers) {
      const [lx, ly] = L.inv ? applyAffine(L.inv, X, Y) : [X, Y];
      const px = Math.round(lx), py = Math.round(ly);
      if (px < 0 || py < 0 || px >= L.rect.w || py >= L.rect.h) continue;
      const v = L.data[(L.rect.y + py) * L.sw + L.rect.x + px];
      const name = scene.multi ? `${channelInfo(L.file.scope, L.layer.channel).label} ` : '';
      if (!lines.length) lines.push(`x ${px}  y ${py}${scene.sensor ? '' : `  (${L.rect.x + px}, ${L.rect.y + py})`}`);
      lines.push(`${name}${fmtBrightness(v)}`);
    }
    if (!lines.length) { this.clearInspect(); return; }
    S.inspect = { sx, sy, lines, key: frameKey(scene.baseFile) };
    this.schedule();
  }

  clearInspect() {
    if (!S.inspect) return;
    S.inspect = null;
    bump('overlay');
  }
}

/** The page's right-hand gutter in pixels (wider beside a notch in landscape). */
function sideGutter(host) {
  const probe = document.createElement('div');
  probe.style.cssText = 'position:absolute;visibility:hidden;height:0;width:var(--gutter-r, 16px)';
  host.appendChild(probe);
  const w = probe.getBoundingClientRect().width;
  probe.remove();
  return w > 0 ? w : 16;
}

/** Round values to label between the ends of a colour scale `px` pixels long. */
function scaleTicks(lo, hi, log, px) {
  if (!(hi > lo)) return [];
  if (!log) return niceTicks(lo, hi, Math.max(1, Math.floor(px / 52))).filter((v) => v > lo && v < hi);
  const out = [];
  const a = lo > 0 ? lo : 1;
  for (let d = Math.floor(Math.log10(a)); d <= Math.ceil(Math.log10(hi)); d++) {
    for (const m of [1, 2, 5]) { const v = m * Math.pow(10, d); if (v > lo && v < hi) out.push(v); }
  }
  return out;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
