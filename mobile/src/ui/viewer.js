// The image canvas: draws the composed scene plus vector overlays (fits, tracks,
// scale bar, colour bar), and turns touches into pan / pinch-zoom / tap-select /
// double-tap-fit / press-and-hold pixel read-out. A tap on the colour bar opens
// the editor for its min and max.

import { composeScene, hueCss, makeNorm } from '../core/render.js';
import { applyAffine } from '../core/geometry.js';
import { SCOPES, channelRect, channelInfo } from '../core/layouts.js';
import { trackColor } from '../core/tracking.js';
import {
  S, subscribe, bump, buildScene, sceneFits, probeAt, select, frameKey, classify, CLASS_COLORS, summarize, brightnessSets,
  setScaleEditor,
} from '../store.js';
import { THEME, fmtBrightness } from './theme.js';
import { niceTicks, fmtAxis } from './plots.js';

const TAP_SLOP = 9, DOUBLE_MS = 330, DOUBLE_DIST = 30, HOLD_MS = 420;

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
    this.edge = 12; this.edgeL = 12;   // clearance from the right and left edges (more beside a notch)
    this.cbar = null;             // where the colour bar was last drawn
    this.ramps = new Map();       // LUT -> its strips as small canvases

    this.onDown = this.onDown.bind(this); this.onMove = this.onMove.bind(this); this.onUp = this.onUp.bind(this);
    const c = this.canvas;
    c.addEventListener('pointerdown', this.onDown);
    c.addEventListener('pointermove', this.onMove);
    c.addEventListener('pointerup', this.onUp);
    c.addEventListener('pointercancel', this.onUp);
    c.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    // A tap here can open a card under the finger. Without this the browser would follow the touch
    // with a click at the same spot, which would land on that card (and could press its Close).
    c.addEventListener('touchend', (e) => { if (e.cancelable) e.preventDefault(); }, { passive: false });
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
    cancelAnimationFrame(this.raf);
    this.canvas.remove();
  }

  // ---- geometry -----------------------------------------------------------

  resize() {
    const r = this.host.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    const first = this.cssW <= 1;
    this.edge = Math.max(12, sideGutter(this.host, 'r') - 4);
    this.edgeL = Math.max(12, sideGutter(this.host, 'l') - 4);
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
    const cx = this.cssW / 2, cy = this.cssH / 2;
    // A view showing the whole image keeps showing it; a zoomed view keeps its centre.
    const keep = this.scene && !first && !this.isFitted() ? this.toScene(cx, cy) : null;
    this.cssW = w; this.cssH = h; this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
    if (keep && this.scene) {
      // keep the same image point in the middle of the canvas
      const [sx, sy] = this.toScreen(keep[0], keep[1]);
      this.view.tx += w / 2 - sx; this.view.ty += h / 2 - sy;
      this.clampView();
    } else if (this.scene) this.fitView();
    this.schedule();
  }

  fitZoom() {
    if (!this.scene) return 1;
    return Math.min(this.cssW / this.scene.w, this.cssH / this.scene.h) * 0.985;
  }

  fitView() {
    if (!this.scene) return;
    const z = this.fitZoom();
    this.view = { z, tx: (this.cssW - this.scene.w * z) / 2, ty: (this.cssH - this.scene.h * z) / 2 };
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
    const v = this.view, fit = this.fitZoom();
    v.z = Math.min(Math.max(v.z, fit * 0.5), Math.max(64, fit * 4));
    const w = this.scene.w * v.z, h = this.scene.h * v.z, m = 48;
    v.tx = Math.min(this.cssW - m, Math.max(m - w, v.tx));
    v.ty = Math.min(this.cssH - m, Math.max(m - h, v.ty));
  }

  zoomAt(sx, sy, factor) {
    if (!this.scene) return;
    const v = this.view;
    const ix = (sx - v.tx) / v.z, iy = (sy - v.ty) / v.z;
    v.z *= factor;
    this.clampView();
    v.tx = sx - ix * v.z; v.ty = sy - iy * v.z;
    this.clampView();
    this.schedule();
  }

  zoomBy(factor) { this.zoomAt(this.cssW / 2, this.cssH / 2, factor); }

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
    if (prevId !== this.sceneId) this.fitView();
  }

  draw() {
    if (this.needCompose) this.compose();
    const ctx = this.ctx, dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = THEME.canvas;
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    const scene = this.scene;
    if (!scene) return;
    const v = this.view;
    ctx.imageSmoothingEnabled = v.z * dpr < 1;
    ctx.drawImage(this.buf, v.tx, v.ty, scene.w * v.z, scene.h * v.z);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (scene.sensor && S.display.showGrid) this.drawChannelGrid(ctx, scene);
    if (S.movie.showTracks) this.drawTracks(ctx, scene);
    this.fits = S.display.showFits ? sceneFits(scene) : [];
    this.drawFits(ctx, scene);
    if (S.display.showScaleBar) this.drawScaleBar(ctx, scene);
    this.cbar = S.display.showColorbar ? this.colorbarBox(scene) : null;
    if (this.cbar) this.drawColorbar(ctx, scene, this.cbar);
    if (S.inspect) this.drawInspect(ctx);
  }

  // ---- colour bar -------------------------------------------------------------

  /**
   * Where the colour bar goes, in CSS pixels: one strip per layer, placed where
   * the fitted image leaves the stage empty. A frame limited by the stage's
   * width leaves room above it, so the bar runs across the top; one limited by
   * the height leaves room beside it, so the bar runs down the right-hand side.
   * Null when the stage is too small for a bar.
   */
  colorbarBox(scene) {
    const n = scene.layers.length;
    const horizontal = this.cssW / scene.w <= this.cssH / scene.h;
    const thick = n > 1 ? (horizontal ? 5 : 7) : 10, gap = n > 1 && horizontal ? 2 : 3;
    const span = n * thick + (n - 1) * gap;
    if (horizontal) {
      // Overlay strips leave room at the right for the unit, which sits beside them.
      const x = this.edgeL, len = this.cssW - this.edgeL - this.edge - (n > 1 ? 30 : 0);
      if (len < 120 || this.cssH < 90) return null;
      return { horizontal, x, y: 12, len, thick, gap, span, n };
    }
    const y = 30, len = Math.min(200, this.cssH - y - 44);      // stays clear of the scale bar
    if (len < 36) return null;
    return { horizontal, x: this.cssW - this.edge - span, y, len, thick, gap, span, n };
  }

  /** True when a touch at (sx, sy) is on the colour bar or its labels. */
  inColorbar(sx, sy) {
    const b = this.cbar;
    if (!b) return false;
    if (b.horizontal) return sy <= b.y + b.span + (b.n > 1 ? 8 : 26);
    return sx >= b.x - 40 && sy >= b.y - 24 && sy <= b.y + b.len + 12;
  }

  /** A LUT as a 256-pixel strip, brightest last (horizontal) or first (vertical). */
  ramp(lut, horizontal) {
    const key = horizontal ? 'h' : 'v';
    let pair = this.ramps.get(lut);
    if (!pair) { pair = {}; this.ramps.set(lut, pair); }
    if (!pair[key]) {
      const c = document.createElement('canvas');
      c.width = horizontal ? 256 : 1; c.height = horizontal ? 1 : 256;
      const g = c.getContext('2d'), img = g.createImageData(c.width, c.height);
      for (let i = 0; i < 256; i++) {
        const k = (horizontal ? i : 255 - i) * 3;
        img.data[i * 4] = lut[k]; img.data[i * 4 + 1] = lut[k + 1]; img.data[i * 4 + 2] = lut[k + 2]; img.data[i * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      pair[key] = c;
    }
    return pair[key];
  }

  /**
   * Colour scale in pps. One image: a strip labelled with its min, its max and
   * round values between them. An overlay: one strip per layer in the layer's
   * colour, without numbers (the editor that a tap opens lists each range).
   */
  drawColorbar(ctx, scene, b) {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    scene.layers.forEach((L, i) => {
      const off = i * (b.thick + b.gap);
      const x = b.horizontal ? b.x : b.x + off, y = b.horizontal ? b.y + off : b.y;
      const w = b.horizontal ? b.len : b.thick, h = b.horizontal ? b.thick : b.len;
      ctx.drawImage(this.ramp(L.lut, b.horizontal), x, y, w, h);
      ctx.lineWidth = 1;
      ctx.strokeStyle = S.scale ? THEME.accent : 'rgba(255,255,255,0.6)';
      ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
    });
    const text = (str, x, y) => {
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(8,10,15,0.8)';
      ctx.strokeText(str, x, y);
      ctx.fillStyle = '#ffffff'; ctx.fillText(str, x, y);
    };
    const tick = (x0, y0, x1, y1) => {
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    };
    const unit = S.display.log ? 'pps, log' : 'pps';
    ctx.font = `600 10.5px ${THEME.mono}`;
    ctx.textBaseline = 'middle';
    const single = scene.layers.length === 1 ? scene.layers[0] : null;
    const norm = single ? makeNorm(single.lo, single.hi, single.log) : null;
    const inner = single ? scaleTicks(single.lo, single.hi, single.log, b.len) : [];
    if (b.horizontal) {
      const ty = b.y + b.span + 13, x1 = b.x + b.len;
      if (!single) { ctx.textAlign = 'left'; text('pps', x1 + 7, b.y + b.span / 2); ctx.restore(); return; }
      const loText = fmtAxis(single.lo), hiText = `${fmtAxis(single.hi)} ${unit}`;
      ctx.textAlign = 'left'; text(loText, b.x, ty);
      ctx.textAlign = 'right'; text(hiText, x1, ty);
      tick(b.x + 0.5, b.y + b.span, b.x + 0.5, b.y + b.span + 4);
      tick(x1 - 0.5, b.y + b.span, x1 - 0.5, b.y + b.span + 4);
      const taken = [[b.x, b.x + ctx.measureText(loText).width], [x1 - ctx.measureText(hiText).width, x1]];
      ctx.textAlign = 'center';
      for (const v of inner) {
        const x = b.x + norm.fwd(v) * b.len, str = fmtAxis(v), half = ctx.measureText(str).width / 2 + 7;
        if (taken.some(([a, c]) => x + half > a && x - half < c)) continue;
        taken.push([x - half + 7, x + half - 7]);
        tick(Math.round(x) + 0.5, b.y + b.span, Math.round(x) + 0.5, b.y + b.span + 4);
        text(str, x, ty);
      }
    } else {
      ctx.textAlign = 'right';
      text(unit, b.x + b.span + 1, b.y - 12);
      if (single) {
        const marks = [{ v: single.hi, y: b.y, ty: b.y + 5 }, { v: single.lo, y: b.y + b.len, ty: b.y + b.len - 5 }];
        for (const v of inner) {
          const y = b.y + (1 - norm.fwd(v)) * b.len;
          if (marks.every((m) => Math.abs(m.ty - y) >= 15)) marks.push({ v, y, ty: y });
        }
        for (const m of marks) {
          const ly = Math.min(b.y + b.len - 0.5, Math.max(b.y + 0.5, Math.round(m.y) + 0.5));
          tick(b.x - 4, ly, b.x, ly);
          text(fmtAxis(m.v), b.x - 7, m.ty);
        }
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
    const target = (this.cssW * 0.22) / z * pix;                    // micrometres
    const k = Math.pow(10, Math.floor(Math.log10(target)));
    const um = [5, 2, 1].map((m) => m * k).find((val) => val <= target) || k;
    const len = (um / pix) * z;
    const x1 = this.cssW - this.edge - 2, y = this.cssH - 16;
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

  onDown(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try { this.canvas.setPointerCapture(e.pointerId); } catch (err) { /* older WebKit */ }
    // Touching the image finishes any number being typed (and puts the keyboard away).
    const active = document.activeElement;
    if (active && active !== document.body && typeof active.blur === 'function') active.blur();
    const p = this.pos(e);
    this.pointers.set(e.pointerId, p);
    clearTimeout(this.holdTimer);
    if (this.pointers.size === 1) {
      // A touch that starts on the colour bar opens its editor if it ends as a tap; a drag still pans.
      const onBar = this.inColorbar(p.x, p.y);
      this.g = { mode: 'pending', onBar, x0: p.x, y0: p.y, lx: p.x, ly: p.y, t0: performance.now() };
      if (!onBar) {
        this.holdTimer = setTimeout(() => {
          if (this.g && this.g.mode === 'pending') { this.g.mode = 'inspect'; this.inspectAt(this.g.lx, this.g.ly); }
        }, HOLD_MS);
      }
    } else if (this.pointers.size === 2) {
      this.clearInspect();
      this.startPinch();
    }
    e.preventDefault();
  }

  startPinch() {
    const [a, b] = Array.from(this.pointers.values());
    this.g = {
      mode: 'pinch', d0: Math.max(10, Math.hypot(a.x - b.x, a.y - b.y)),
      mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, view0: { ...this.view },
    };
  }

  onMove(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const p = this.pos(e);
    this.pointers.set(e.pointerId, p);
    const g = this.g;
    if (!g || !this.scene) return;
    if (g.mode === 'pinch' && this.pointers.size >= 2) {
      const [a, b] = Array.from(this.pointers.values());
      const d = Math.max(10, Math.hypot(a.x - b.x, a.y - b.y));
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const v0 = g.view0;
      const ix = (g.mx - v0.tx) / v0.z, iy = (g.my - v0.ty) / v0.z;
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
      this.view.tx += p.x - g.lx; this.view.ty += p.y - g.ly;
      this.clampView();
      this.schedule();
    } else if (g.mode === 'inspect') {
      this.inspectAt(p.x, p.y);
    }
    g.lx = p.x; g.ly = p.y;
  }

  onUp(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const p = this.pointers.get(e.pointerId);
    this.pointers.delete(e.pointerId);
    clearTimeout(this.holdTimer);
    const g = this.g;
    if (!g) return;
    if (g.mode === 'pinch') {
      if (this.pointers.size === 1) {
        const q = Array.from(this.pointers.values())[0];
        this.g = { mode: 'pan', x0: q.x, y0: q.y, lx: q.x, ly: q.y, t0: performance.now() };
      } else if (this.pointers.size === 0) this.g = null;
      return;
    }
    this.g = null;
    if (g.mode === 'inspect') { this.clearInspect(); return; }
    if (g.mode === 'pending' && e.type === 'pointerup' && this.scene) {
      if (g.onBar) { this.lastTap = null; setScaleEditor(!S.scale); } else this.onTap(p.x, p.y);
    }
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

/** The page's left or right gutter in pixels (wider beside a notch in landscape). */
function sideGutter(host, side) {
  const probe = document.createElement('div');
  probe.style.cssText = `position:absolute;visibility:hidden;height:0;width:var(--gutter-${side}, 16px)`;
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
