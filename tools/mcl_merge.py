"""
MCL channel merge (used by Batch Convert).

The MCL images four emission bands side by side on one camera frame, one per
quadrant. Region numbering matches utils.integrate_sif, displayed origin='lower':

    ┌──────┬───────┐
    │ Blue │ Green │      1 │ 2
    ├──────┼───────┤
    │ Red  │  NIR  │      3 │ 4
    └──────┴───────┘

Each band goes through its own filter path, so the quadrants are not pixel
aligned: a particle sits at a slightly different place (offset, and to a lesser
degree scale/rotation) in every channel. This module splits a frame into
channels, maps each channel onto a reference channel with an affine transform,
and colour-composites the result.

Registration sources, best first:
  * a registration image: particles that emit in every channel. Spots are
    matched across channels and a full affine is fit (RANSAC, so smeared rows
    and one-channel spots don't bias it).
  * the data itself: the same matching, pooled over every uploaded file (the
    channel geometry is a property of the optics, not of the field). Works
    whenever some particles show up in both channels.
  * the field-stop aperture: the translation that overlays the fitted aperture
    circles. Only possible where the aperture is visible, which in practice is
    the bright channels.
"""
import io
import os
import json
import shutil
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import date

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

MCL_PIX_UM = 0.1011
CHANNELS = ("Blue", "Green", "Red", "NIR")

# Default colours: each stays recognisably its own name (NIR, having no colour,
# is pink as "past red"), every one is bright on black (CIE L* >= 55), and the
# set was chosen to maximise the smallest pairwise CIEDE2000 difference under
# normal, protan, deutan and tritan vision (Machado 2009 simulation). Worst
# case ~20 dE, against ~19 for pure RGB + grey, whose blue is too dark to see
# on black (L* 32) and whose grey NIR reads as colocalisation white.
PALETTES = {
    "Intuitive, colorblind-safe": {"Blue": "#597AFF", "Green": "#33FF85",
                                   "Red": "#FF2A00", "NIR": "#FF80AC"},
    "Classic RGB": {"Blue": "#0000FF", "Green": "#00FF00",
                    "Red": "#FF0000", "NIR": "#FFFFFF"},
    # Assigned by position in the selection, not by channel name.
    "Green / Magenta (2 channels)": ("#00FF66", "#FF33FF"),
}

REG_IMAGE = "Registration image"
REG_AUTO = "Auto (particles in data, then aperture)"
REG_APERTURE = "Aperture only"
REG_NONE = "None (raw quadrants)"


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------
def load_sif_cps(path):
    """First frame of a .sif in photons/s, normalised exactly as integrate_sif."""
    import sif_parser
    data, meta = sif_parser.np_open(path, ignore_corrupt=True)
    frame = np.asarray(data[0], dtype=float)
    gain = meta.get("GainDAC") or 1  # gain turned off reads 0
    return frame * (5.0 / gain) / meta["ExposureTime"] / meta["AccumulatedCycles"]


@st.cache_data(show_spinner=False, max_entries=128)
def _load_upload(name, data):
    # sif_parser needs a real file (a BytesIO reads as zero frames), so spill
    # to a per-call scratch dir, as process_files does.
    tmp = tempfile.mkdtemp(prefix="psyfit_mcl_")
    try:
        path = os.path.join(tmp, os.path.basename(name))
        with open(path, "wb") as f:
            f.write(data)
        return load_sif_cps(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_upload(uploaded_file):
    return _load_upload(uploaded_file.name, uploaded_file.getvalue())


def split_channels(frame):
    """Quadrants of a full MCL frame (midpoint split), keyed by channel name."""
    h, w = frame.shape
    r, c = h // 2, w // 2
    return {
        "Blue": frame[r:2 * r, :c],
        "Green": frame[r:2 * r, c:2 * c],
        "Red": frame[:r, :c],
        "NIR": frame[:r, c:2 * c],
    }


# ---------------------------------------------------------------------------
# Spot detection
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=1024)
def detect_spots(img, k=6.0, min_distance=4, max_spots=300):
    """
    Sub-pixel (x, y) positions of bright point-like spots. The image is
    high-passed (light smoothing minus a 15 px median, which takes out the
    aperture and illumination profile) and thresholded at k robust sigmas.
    """
    img = np.asarray(img, dtype=float)
    img = np.where(np.isfinite(img), img, np.nanmedian(img))
    hp = ndimage.gaussian_filter(img, 1.0) - ndimage.median_filter(img, size=15)
    # Saturated frames smear into full-width horizontal bands; a row's median
    # is background plus band, so subtracting it leaves the particles.
    hp -= np.median(hp, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    if not noise > 0:
        return np.empty((0, 2))
    peaks = peak_local_max(hp, min_distance=min_distance, threshold_abs=k * noise,
                           exclude_border=6, num_peaks=max_spots)
    pos = np.clip(hp, 0, None)
    dy, dx = np.mgrid[-2:3, -2:3]
    out = []
    for r, c in peaks:
        win = pos[r - 2:r + 3, c - 2:c + 3]
        s = win.sum()
        if s > 0:
            out.append((c + (win * dx).sum() / s, r + (win * dy).sum() / s))
    return np.array(out, dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Transform estimation. Matrices are 3x3 homogeneous, acting on (x, y) =
# (column, row), and map a channel's coordinates onto the reference channel's.
# ---------------------------------------------------------------------------
def _apply(m, pts):
    return pts @ m[:2, :2].T + m[:2, 2]


def _translation(t):
    m = np.eye(3)
    m[:2, 2] = t
    return m


def _lstsq_affine(src, dst):
    a = np.column_stack([src, np.ones(len(src))])
    sol, *_ = np.linalg.lstsq(a, dst, rcond=None)
    m = np.eye(3)
    m[:2, :] = sol.T
    return m


def _ransac_affine(src, dst, thr, trials=600, seed=0):
    """Affine from 3-point samples; returns (matrix, inlier mask) refit on inliers."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best = np.zeros(n, bool)
    for _ in range(trials):
        idx = rng.choice(n, 3, replace=False)
        # Skip near-collinear samples (ill-conditioned).
        v1, v2 = src[idx[1]] - src[idx[0]], src[idx[2]] - src[idx[0]]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 25.0:
            continue
        m = _lstsq_affine(src[idx], dst[idx])
        inl = np.linalg.norm(_apply(m, src) - dst, axis=1) < thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None, best
    return _lstsq_affine(src[best], dst[best]), best


def affine_params(m):
    """Readable decomposition: translation, rotation (deg), x/y scale, shear (deg)."""
    a = m[:2, :2]
    sx = np.hypot(a[0, 0], a[1, 0])
    rot = np.arctan2(a[1, 0], a[0, 0])
    shear = np.arctan2(a[0, 1], a[1, 1]) + rot
    sy = np.linalg.det(a) / sx if sx else 0.0
    return {"dx": m[0, 2], "dy": m[1, 2], "rotation_deg": np.degrees(rot),
            "scale_x": sx, "scale_y": sy, "shear_deg": np.degrees(shear)}


def _plausible(m):
    # The channels see the same field through near-identical optics: anything
    # beyond a few % scale or a few degrees is a bad match, not real geometry.
    p = affine_params(m)
    return (0.93 < p["scale_x"] < 1.07 and 0.93 < p["scale_y"] < 1.07
            and abs(p["rotation_deg"]) < 4 and abs(p["shear_deg"]) < 3)


def _vote_translation(pairs, max_shift):
    """Most common displacement (ref - mov) over all spot pairs, pooled over images."""
    nb = 2 * max_shift + 1
    hist = np.zeros((nb, nb))
    edges = np.arange(-max_shift - 0.5, max_shift + 1.0)
    for ref, mov in pairs:
        if len(ref) and len(mov):
            d = (ref[None, :, :] - mov[:, None, :]).reshape(-1, 2)
            d = d[(np.abs(d) <= max_shift).all(axis=1)]
            hist += np.histogram2d(d[:, 1], d[:, 0], bins=[edges, edges])[0]
    hist = ndimage.gaussian_filter(hist, 1.0)
    iy, ix = np.unravel_index(np.argmax(hist), hist.shape)
    return np.array([ix - max_shift, iy - max_shift], dtype=float)


def _match(ref, mov, m, tol):
    """Mutual-nearest-neighbour pairs within `tol` px once `mov` is mapped by `m`."""
    if not len(ref) or not len(mov):
        return np.empty((0, 2)), np.empty((0, 2))
    mapped = _apply(m, mov)
    d, j = cKDTree(ref).query(mapped, distance_upper_bound=tol)
    back = cKDTree(mapped).query(ref)[1]
    i = np.flatnonzero(np.isfinite(d))
    i = i[back[j[i]] == i]
    return mov[i], ref[j[i]]


@dataclass
class Registration:
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    method: str = "identity"      # reference | particles | aperture | identity
    model: str = ""               # affine | translation
    n_matches: int = 0
    rms_px: float = float("nan")
    note: str = ""


def register_by_particles(pairs, *, max_shift=48, area_px=256 * 256):
    """
    Affine (or, short of matches, translation) mapping `mov` spots onto `ref`
    spots, pooled over every (ref_spots, mov_spots) pair. None when the matches
    can't be told apart from chance.
    """
    pairs = [(np.asarray(r), np.asarray(m)) for r, m in pairs]
    if not any(len(r) and len(m) for r, m in pairs):
        return None
    m = _translation(_vote_translation(pairs, max_shift))
    model = "translation"
    src = dst = np.empty((0, 2))
    for tol in (5.0, 3.0, 2.0):
        matched = [_match(r, mv, m, tol) for r, mv in pairs]
        src = np.concatenate([s for s, _ in matched])
        dst = np.concatenate([d for _, d in matched])
        if len(src) == 0:
            return None
        fit = None
        if len(src) >= 6:
            fit, inl = _ransac_affine(src, dst, thr=min(tol, 2.0))
            if fit is not None and inl.sum() >= 6 and _plausible(fit):
                src, dst = src[inl], dst[inl]
            else:
                fit = None
        if fit is not None:
            m, model = fit, "affine"
        else:
            m, model = _translation(np.median(dst - src, axis=0)), "translation"

    # Chance matches: spots scattered at random would pair up within 2 px about
    # n_ref * n_mov * pi * 2^2 / area times per image. Demand well above that.
    chance = sum(len(r) * len(mv) for r, mv in pairs) * np.pi * 4.0 / area_px
    if len(src) < max(4, 3 * chance):
        return None
    rms = float(np.sqrt(np.mean(np.sum((_apply(m, src) - dst) ** 2, axis=1))))
    return Registration(m, "particles", model, int(len(src)), rms)


def _lowpass(img):
    img = np.where(np.isfinite(img), img, np.nanmedian(img))
    return ndimage.gaussian_filter(ndimage.median_filter(img, size=9), 3)


@st.cache_data(show_spinner=False, max_entries=1024)
def aperture_shift(ref_img, mov_img, max_shift=48):
    """
    Translation (x, y) carrying `mov_img` onto `ref_img`, from the illuminated
    aperture: cross-correlation of the low-passed images (particles median-
    filtered out, leaving the field-stop outline and illumination profile).

    Checked against particle registration on Tm/Yb UCNP data this lands within
    ~0.5 px, where fitting a circle to each aperture was ~5 px off (the
    aperture is clipped differently in each channel, which biases a circle fit
    but not a correlation of the whole outline). Returns None when the
    estimate is not trustworthy: the plain and edge-filtered correlations
    disagree (no common outline to lock onto), or the shift is implausibly big.
    """
    from skimage.filters import sobel
    from skimage.registration import phase_cross_correlation
    a, b = _lowpass(ref_img), _lowpass(mov_img)
    plain = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)[0]
    edges = phase_cross_correlation(sobel(a), sobel(b), upsample_factor=10,
                                    normalization=None)[0]
    if np.hypot(*(plain - edges)) > 1.5 or np.abs(plain).max() > max_shift:
        return None
    return plain[::-1].astype(float)  # (row, col) -> (x, y)


def register_by_aperture(ref_imgs, mov_imgs, max_shift=48):
    """Translation from the aperture outline, the median over images that agree."""
    shifts = [aperture_shift(r, m, max_shift) for r, m in zip(ref_imgs, mov_imgs)]
    shifts = np.array([v for v in shifts if v is not None])
    if not len(shifts):
        return None
    med = np.median(shifts, axis=0)
    agree = np.hypot(*(shifts - med).T) <= 2.0
    # Images disagreeing among themselves means no reliable common outline.
    if agree.sum() < max(1, 0.5 * len(shifts)):
        return None
    t = np.median(shifts[agree], axis=0)
    spread = float(np.sqrt(np.mean(np.sum((shifts[agree] - t) ** 2, axis=1))))         if agree.sum() > 1 else float("nan")
    return Registration(_translation(t), "aperture", "translation", int(agree.sum()), spread,
                        note=f"aperture outline agreed in {agree.sum()}/{len(ref_imgs)} image(s)")


def refine_per_file(ref_spots, mov_spots, m, tol=2.5, min_matches=5):
    """
    Small per-file translation on top of `m` from that file's own matched
    particles (alignment drifts between sessions, so a registration image
    from another session can be off by a pixel or two). Returns (matrix,
    n_matches); `m` unchanged when too few particles match.
    """
    src, dst = _match(ref_spots, mov_spots, m, tol)
    if len(src) < min_matches:
        return m, 0
    return _translation(np.median(dst - _apply(m, src), axis=0)) @ m, int(len(src))


def register_channels(sources, reference, channels, mode, aperture_sources=None,
                      max_shift=48):
    """
    One Registration per channel in `channels`, onto `reference`.

    `sources` is a list of split frames ({channel: image}) to match particles in:
    the registration image for REG_IMAGE, the uploaded data for REG_AUTO.
    `aperture_sources` (default `sources`) are the frames used for the aperture
    fallback. Each channel falls back particles -> aperture -> identity.
    """
    aperture_sources = aperture_sources or sources
    out = {}
    for ch in channels:
        if ch == reference:
            out[ch] = Registration(method="reference")
            continue
        if mode == REG_NONE:
            out[ch] = Registration(note="registration off")
            continue
        reg = None
        if mode in (REG_IMAGE, REG_AUTO):
            pairs = [(detect_spots(s[reference]), detect_spots(s[ch])) for s in sources]
            reg = register_by_particles(pairs, max_shift=max_shift,
                                        area_px=sources[0][reference].size)
        if reg is None and mode != REG_NONE:
            reg = register_by_aperture([s[reference] for s in aperture_sources],
                                       [s[ch] for s in aperture_sources], max_shift)
            if reg is not None and mode != REG_APERTURE:
                reg.note = "no particle matches; " + reg.note
        out[ch] = reg or Registration(note="no particle matches and no aperture "
                                           "visible in both channels; left unaligned")
    return out


def warp_to_reference(img, m, shape):
    """Resample a channel into the reference frame; pixels from outside its quadrant are NaN."""
    if np.allclose(m, np.eye(3)) and img.shape == tuple(shape):
        return np.asarray(img, dtype=float)
    # ndimage maps output (row, col) -> input (row, col): invert, then swap axes.
    inv = np.linalg.inv(m)
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    inv_rc = swap @ inv @ swap
    return ndimage.affine_transform(np.asarray(img, dtype=float), inv_rc[:2, :2],
                                    offset=inv_rc[:2, 2], output_shape=tuple(shape),
                                    order=1, cval=np.nan)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_limits(imgs, saturate_pct, black_sigma=2.0):
    """Default (vmin, vmax) in raw pps for a channel: vmin `black_sigma` robust
    noise sigmas above the median background (so noise speckle drops to black),
    vmax so `saturate_pct` % of pixels clip. Pooled over `imgs`."""
    vals = np.concatenate([np.asarray(i)[np.isfinite(i)].ravel() for i in imgs])
    if vals.size == 0:
        return 0.0, 1.0
    med, hi = np.percentile(vals, [50.0, 100.0 - saturate_pct])
    sigma = 1.4826 * np.median(np.abs(vals - med))
    lo = min(med + black_sigma * sigma, med + 0.5 * (hi - med))
    return float(lo), float(hi if hi > lo else lo + 1.0)


def composite(chans, colors, limits, gamma=1.0):
    """Additive colour composite (Fiji 'Composite' style), clipped to [0, 1]."""
    shape = next(iter(chans.values())).shape
    rgb = np.zeros(shape + (3,))
    for ch, img in chans.items():
        lo, hi = limits[ch]
        v = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0, 1) ** gamma
        rgb += v[..., None] * np.asarray(to_rgb(colors[ch]))
    return np.clip(rgb, 0, 1)


def channel_colors(palette, selected, custom=None):
    if palette == "Green / Magenta (2 channels)":
        pair = PALETTES[palette]
        base = PALETTES["Intuitive, colorblind-safe"]
        cols = {ch: (pair[i] if i < 2 else base[ch]) for i, ch in enumerate(selected)}
    else:
        cols = {ch: PALETTES[palette][ch] for ch in selected}
    cols.update({k: v for k, v in (custom or {}).items() if k in cols})
    return cols


def _add_scalebar(ax, shape, pix_um, length_um):
    h, w = shape
    L = length_um / pix_um
    x0, y0 = w - 0.06 * w - L, 0.06 * h  # bottom right (origin='lower')
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 + 0.03 * h, f"{length_um:g} µm", color="white",
            ha="center", va="bottom", fontsize=8)


def _channel_colorbar(fig, cax, ch, color, lo, hi, gamma, unit):
    """Vertical black -> `color` bar mapping values exactly as composite() does."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from matplotlib.ticker import EngFormatter, MaxNLocator
    cmap = LinearSegmentedColormap.from_list(ch, ["black", color])
    cb = fig.colorbar(ScalarMappable(PowerNorm(gamma, vmin=lo, vmax=hi), cmap), cax=cax,
                      orientation="vertical")
    cb.locator = MaxNLocator(4)
    cb.formatter = EngFormatter(sep="")  # 1.5k, 2M: no offset text to collide
    cb.update_ticks()
    cb.set_label(f"{ch} ({unit})", color=color, fontsize=7, labelpad=2)
    cb.outline.set_edgecolor("white")
    cax.tick_params(colors="white", labelsize=6, length=2, pad=1)


# Panel geometry (inches): image width, and the strip each colorbar takes
# beside it (bar + tick labels + rotated label).
_PANEL_W, _BAR_STRIP_W, _BAR_W = 3.6, 0.62, 0.11


def merge_grid_figure(items, colors, pix_um=None, scalebar_um=None, ncols=4,
                      limits=None, gamma=1.0, unit="pps"):
    """
    Grid of composites; `items` is a list of (title, rgb). With `limits`
    ({channel: (black, vmax)}, shared by every panel) each panel gets one
    vertical colorbar per channel beside it; otherwise a channel legend sits on
    top.
    """
    n = len(items)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    strip = _BAR_STRIP_W * len(colors) if limits else 0.0
    legend_h = 0.0 if limits else 0.5
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, facecolor="black",
                             figsize=((_PANEL_W + strip) * ncols, 4.1 * nrows + legend_h))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, rgb) in zip(axes.flat, items):
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title("\n".join(textwrap.wrap(title, 32)), color="white", fontsize=8)
        if pix_um and scalebar_um:
            _add_scalebar(ax, rgb.shape[:2], pix_um, scalebar_um)
        if limits:
            # Bars live in the image's own axes coordinates, so they always span
            # exactly the image height; the image is pinned left of its cell to
            # leave the strip on the right for them.
            ax.set_anchor("W")
            for j, (ch, col) in enumerate(colors.items()):
                x = 1 + (0.06 + j * _BAR_STRIP_W) / _PANEL_W
                lo, hi = limits[ch]
                _channel_colorbar(fig, ax.inset_axes([x, 0, _BAR_W / _PANEL_W, 1]),
                                  ch, col, lo, hi, gamma, unit)
    if not limits:
        handles = [plt.Line2D([], [], color=c, lw=6) for c in colors.values()]
        leg = fig.legend(handles, list(colors), loc="upper center", ncol=len(colors),
                         frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
        for t, c in zip(leg.get_texts(), colors.values()):
            t.set_color(c)
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_h / (4.1 * nrows + legend_h)))
    return fig


def registration_table(regs, pix_um):
    rows = []
    for ch, r in regs.items():
        p = affine_params(r.matrix)
        rows.append({
            "channel": ch, "method": r.method, "model": r.model or "-",
            "matches": r.n_matches if r.method in ("particles", "aperture") else None,
            "rms (px)": None if np.isnan(r.rms_px) else round(r.rms_px, 2),
            "rms (nm)": None if np.isnan(r.rms_px) else round(r.rms_px * pix_um * 1000, 0),
            "dx (px)": round(p["dx"], 2), "dy (px)": round(p["dy"], 2),
            "rotation (°)": round(p["rotation_deg"], 3),
            "scale x": round(p["scale_x"], 4), "scale y": round(p["scale_y"], 4),
            "note": r.note,
        })
    return rows


def _fig_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt.lower(), bbox_inches="tight", dpi=300,
                facecolor=fig.get_facecolor())
    return buf.getvalue()


MIME = {"svg": "image/svg+xml", "tiff": "image/tiff", "png": "image/png", "jpeg": "image/jpeg"}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_inputs():
    """Merge-mode controls for the Batch Convert input column; returns settings."""
    s = {}
    s["channels"] = st.multiselect("Channels to overlay", CHANNELS, default=list(CHANNELS),
                                   key="mcl_channels")
    s["reg_mode"] = st.selectbox(
        "Registration", [REG_IMAGE, REG_AUTO, REG_APERTURE, REG_NONE], key="mcl_reg_mode",
        help="How channels are aligned onto the reference channel:  \n"
             "- **Registration image**: a .sif of particles that emit in every channel "
             "(e.g. Tm/Yb UCNPs). Spots are matched across quadrants and a full affine "
             "(shift, scale, rotation, shear) is fit. Most accurate; take it in the same "
             "session as the data, since the alignment shifts between sessions.  \n"
             "- **Auto**: the same matching on the uploaded files themselves, pooled over "
             "the whole batch. Needs some particles visible in both channels; channels "
             "with none fall back to the aperture.  \n"
             "- **Aperture only**: shifts each channel so the illuminated aperture outlines "
             "overlay (cross-correlation; translation only, ~1 px). Needs some of the "
             "outline visible in both channels.  \n"
             "Any channel that can't be registered is left unaligned and flagged.")
    s["reg_file"] = None
    if s["reg_mode"] == REG_IMAGE:
        from utils import file_uploader_with_clear
        s["reg_file"] = file_uploader_with_clear(
            "Registration image (.sif)", key="mcl_reg_upload", type=["sif"],
            help="A field of particles that emit in all channels you overlay.")
    s["refine"] = s["reg_mode"] != REG_NONE and st.checkbox(
        "Per-file fine shift", value=True, key="mcl_refine",
        help="After registering, nudge each channel of each file by the median offset of "
             "its own matched particles (needs 5+ particles visible in both channels; "
             "files without them keep the shared transform). Corrects drift of a pixel or "
             "two, e.g. a registration image from another session.")
    s["reference"] = st.selectbox(
        "Reference channel", CHANNELS, index=0, key="mcl_reference",
        help="Other channels are mapped into this one's pixel grid. Pick a channel with "
             "plenty of bright particles.")
    s["export_format"] = st.selectbox("Export Format", ["PNG", "TIFF", "SVG", "JPEG"],
                                      key="mcl_export_format")

    with st.expander("Colors and contrast", expanded=False):
        s["palette"] = st.selectbox("Palette", list(PALETTES), key="mcl_palette",
                                    help="The default keeps each channel its own color "
                                         "while staying distinguishable with red-green or "
                                         "blue-yellow color blindness. Green / Magenta is "
                                         "the safest choice for exactly two channels.")
        base = channel_colors(s["palette"], s["channels"])
        custom = {}
        if s["channels"]:
            cols = st.columns(len(s["channels"]))
            for col, ch in zip(cols, s["channels"]):
                # Keyed on the palette so switching palette resets the pickers.
                custom[ch] = col.color_picker(ch, base[ch], key=f"mcl_color_{s['palette']}_{ch}")
        s["colors"] = channel_colors(s["palette"], s["channels"], custom)
        s["saturate_pct"] = st.slider("Auto vmax: saturated pixels (%)", 0.01, 2.0, 0.2, 0.01,
                                      key="mcl_saturate",
                                      help="Sets each channel's default vmax: the share of "
                                           "its brightest pixels (over the whole batch) "
                                           "drawn at full color. Higher = dim particles "
                                           "more visible. Changing it refits every vmax to "
                                           "the current batch; edit them per channel above "
                                           "the merged images.")
        s["gamma"] = st.slider("Gamma", 0.3, 1.5, 1.0, 0.05, key="mcl_gamma",
                               help="1 = linear between vmin and vmax. Below 1 lifts dim "
                                    "particles (and background noise) relative to bright "
                                    "ones.")
        sb1, sb2 = st.columns(2)
        s["pix_um"] = sb1.number_input("Pixel size (µm)", value=MCL_PIX_UM, format="%.4f",
                                       key="mcl_pix")
        s["scalebar_um"] = sb2.number_input("Scale bar (µm, 0 = off)", value=5.0,
                                            min_value=0.0, step=1.0, key="mcl_scalebar")
    return s


# vmin/vmax per channel, kept for the whole session so every batch is drawn on
# the same scale. Held under a plain key: Streamlit drops a widget's own state on
# any run where the widget isn't drawn (e.g. while no files are uploaded).
_LIMITS_STORE = "mcl_display_limits"


def _cached_limit(container, key, auto, label, help, reseed=False):
    """number_input whose value survives new uploads. Seeded from `auto` only
    the first time `key` is seen or when `reseed` is set."""
    store = st.session_state.setdefault(_LIMITS_STORE, {})
    if key in st.session_state:          # widget alive: pick up any edit
        store[key] = st.session_state[key]
    if reseed or key not in store:
        store[key] = auto
    st.session_state[key] = store[key]
    value = container.number_input(label, min_value=0.0, format="%.4g", key=key,
                                   help=help + " Kept when you upload new files, so "
                                               "batches stay comparable; use Reset to "
                                               "auto to fit the current batch.")
    store[key] = value
    return value


def _reset_limits():
    st.session_state.pop(_LIMITS_STORE, None)
    for k in [k for k in st.session_state if k.startswith(("mcl_vmin_", "mcl_vmax_"))]:
        del st.session_state[k]


def _split_upload(f):
    frame = load_upload(f)
    if frame.ndim != 2 or frame.shape[0] < 32 or frame.shape[1] < 32:
        raise ValueError(f"{f.name}: unexpected frame shape {frame.shape}")
    return split_channels(frame)


def render_outputs(uploaded_files, s):
    channels = list(s["channels"])
    if not channels:
        st.info("Pick at least one channel to overlay.")
        return
    reference = s["reference"]

    with st.spinner("Loading files..."):
        data = {}
        for f in uploaded_files:
            try:
                data[f.name] = _split_upload(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
    if not data:
        return
    shape = data[next(iter(data))][reference].shape
    bad = [n for n, d in data.items() if d[reference].shape != shape]
    if bad:
        st.error(f"Frame size differs from the first file in: {', '.join(bad)}")
        return

    reg_sources, reg_label = list(data.values()), "the uploaded files"
    if s["reg_mode"] == REG_IMAGE:
        if not s["reg_file"]:
            st.info("Upload a registration image, or choose another registration method.")
            return
        try:
            reg_split = _split_upload(s["reg_file"])
        except Exception as e:
            st.error(f"Could not read the registration image: {e}")
            return
        if reg_split[reference].shape != shape:
            st.error("The registration image's frame size doesn't match the data.")
            return
        reg_sources, reg_label = [reg_split], s["reg_file"].name

    with st.spinner("Registering channels..."):
        regs = register_channels(reg_sources, reference, channels, s["reg_mode"],
                                 aperture_sources=reg_sources + list(data.values())
                                 if s["reg_mode"] == REG_IMAGE else None)

    unaligned = [ch for ch, r in regs.items() if r.method == "identity" and s["reg_mode"] != REG_NONE]
    if unaligned:
        st.warning(f"Could not register {', '.join(unaligned)} onto {reference}; "
                   "shown unaligned. See the registration details below.")

    # Per-file matrices: the shared transform, optionally fine-shifted per file.
    file_mats = {name: {ch: regs[ch].matrix for ch in channels} for name in data}
    if s["refine"]:
        for ch in channels:
            if regs[ch].method in ("reference", "identity"):
                continue
            n_refined = 0
            for name, d in data.items():
                m, n = refine_per_file(detect_spots(d[reference]), detect_spots(d[ch]),
                                       regs[ch].matrix)
                file_mats[name][ch] = m
                n_refined += n > 0
            regs[ch].note = "; ".join(filter(None, [
                regs[ch].note, f"fine-shifted {n_refined}/{len(data)} file(s)"]))

    # Register every file into the reference grid.
    registered = {name: {ch: warp_to_reference(d[ch], file_mats[name][ch], shape)
                         for ch in channels}
                  for name, d in data.items()}
    # One display range per channel in raw pps, shared by every file, so each
    # channel's colorbar holds for every panel.
    unit = "pps"
    auto = {ch: display_limits([v[ch] for v in registered.values()], s["saturate_pct"])
            for ch in channels}
    # Changing the auto-vmax % is an explicit request for new auto vmax values.
    pct_changed = st.session_state.get("_mcl_auto_pct", s["saturate_pct"]) != s["saturate_pct"]
    st.session_state["_mcl_auto_pct"] = s["saturate_pct"]
    st.button("Reset vmin/vmax to auto", key="mcl_reset_limits", on_click=_reset_limits,
              help="Refit every channel's vmin and vmax to the files uploaded now. "
                   "Otherwise they are kept across uploads so batches share a scale.")
    limits = {}
    for col, ch in zip(st.columns(len(channels)), channels):
        lo_auto, hi_auto = auto[ch]
        vmin = _cached_limit(col, f"mcl_vmin_{ch}", lo_auto, f"{ch} vmin ({unit})",
                             f"Drawn black at or below this. Auto: the background median "
                             f"+ 2 noise σ ({lo_auto:.4g} for this batch), which hides "
                             f"camera noise speckle; raise it to hide more.")
        vmax = _cached_limit(col, f"mcl_vmax_{ch}", hi_auto, f"{ch} vmax ({unit})",
                             f"Drawn at full color at or above this. Auto: the value "
                             f"saturating {s['saturate_pct']:g}% of {ch} pixels "
                             f"({hi_auto:.4g} for this batch).", reseed=pct_changed)
        if vmax <= vmin:
            col.caption("vmax must exceed vmin; using vmin + 1.")
            vmax = vmin + 1.0
        limits[ch] = (float(vmin), float(vmax))

    merges = [(name, composite(chans, s["colors"], limits, s["gamma"]))
              for name, chans in registered.items()]

    fig = merge_grid_figure(merges, s["colors"], s["pix_um"], s["scalebar_um"],
                            limits=limits, gamma=s["gamma"], unit=unit)
    st.pyplot(fig)

    fmt = s["export_format"].lower()
    today = date.today().strftime("%Y%m%d")
    d1, d2 = st.columns(2)
    d1.download_button(f"Download grid ({fmt.upper()})", _fig_bytes(fig, fmt),
                       file_name=f"mcl_merge_grid_{today}.{fmt}", mime=MIME[fmt])
    plt.close(fig)
    d2.download_button("Download per-file images + registered stacks (ZIP)",
                       _build_zip(merges, registered, regs, file_mats, s, reference,
                                  reg_label, limits, unit),
                       file_name=f"mcl_merge_{today}.zip", mime="application/zip")

    with st.expander("Registration details", expanded=bool(unaligned)):
        st.caption(f"Reference: **{reference}**. Registered from {reg_label}. "
                   "Transforms map each channel's quadrant pixels onto the reference "
                   "quadrant; rms is the residual of matched spots (particles) or the "
                   "spread of aperture shifts across images (aperture).")
        st.dataframe(registration_table(regs, s["pix_um"]), hide_index=True)
        _render_before_after(reg_sources[0], regs, channels, reference, shape, s)


def _render_before_after(split, regs, channels, reference, shape, s):
    """Before/after composite of the registration source, the visual check."""
    before = {ch: split[ch] for ch in channels}
    after = {ch: warp_to_reference(split[ch], regs[ch].matrix, shape) for ch in channels}
    lim = {ch: display_limits([before[ch]], s["saturate_pct"]) for ch in channels}
    fig = merge_grid_figure([("Before registration", composite(before, s["colors"], lim, s["gamma"])),
                             ("After registration", composite(after, s["colors"], lim, s["gamma"]))],
                            s["colors"], ncols=2)
    st.pyplot(fig)
    plt.close(fig)


def _build_zip(merges, registered, regs, file_mats, s, reference, reg_label, limits, unit):
    import tifffile
    channels = list(s["channels"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, rgb in merges:
            stem = os.path.splitext(os.path.basename(name))[0]
            # Native resolution, flipped so the saved image is oriented as displayed
            # (origin='lower').
            png = io.BytesIO()
            plt.imsave(png, rgb[::-1], format="png")
            z.writestr(f"{stem}_merge.png", png.getvalue())
            stack = np.stack([registered[name][ch][::-1] for ch in channels]).astype(np.float32)
            tif = io.BytesIO()
            tifffile.imwrite(tif, stack, imagej=True,
                             resolution=(1 / s["pix_um"], 1 / s["pix_um"]),
                             metadata={"axes": "CYX", "unit": "um", "Labels": channels})
            z.writestr(f"{stem}_registered.tif", tif.getvalue())
        meta = {
            "reference": reference, "registered_from": reg_label, "mode": s["reg_mode"],
            "pixel_size_um": s["pix_um"],
            "note": "matrix maps (x=col, y=row) in a channel quadrant onto the reference "
                    "quadrant, origin at the quadrant's first array row (displayed bottom-left). "
                    "Registered stacks are photons/s, NaN outside the source quadrant.",
            "channels": {ch: {**{k: v for k, v in asdict(r).items() if k != "matrix"},
                              "rms_px": None if np.isnan(r.rms_px) else r.rms_px,
                              "matrix": np.asarray(r.matrix).tolist(),
                              "color": s["colors"].get(ch)}
                         for ch, r in regs.items()},
            "display": {"unit": unit, "gamma": s["gamma"],
                        "channels": {ch: {"vmin": lo, "vmax": hi}
                                     for ch, (lo, hi) in limits.items()}},
            "per_file_matrices": {name: {ch: np.asarray(m).tolist() for ch, m in mats.items()}
                                  for name, mats in file_mats.items()},
        }
        z.writestr("registration.json", json.dumps(meta, indent=2, default=float))
    return buf.getvalue()
