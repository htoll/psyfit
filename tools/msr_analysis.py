"""Load, inspect, and analyze Abberior Imspector ``.msr`` files.

``.msr`` files are OBF (Omas Binary Format) containers: one file holds many
"stacks" (channels / detection windows / measurements), each with its own pixel
data, physical pixel size, and a block of Imspector acquisition metadata
(excitation wavelength, laser power, imaging window, dwell time, and so on).

This tool wraps :mod:`msr_reader` (a pure-Python OBF reader) and provides:

* **Preview and select**: thumbnail grid of every stack in the file with
  checkboxes to choose which images to carry into analysis/overlay.
* **Metadata**: a collapsed dropdown of *all* saved metadata per stack, with the
  key acquisition parameters (excitation wavelength, imaging window, excitation
  intensity, dwell, pixel size) surfaced at the top.
* **Confocal brightness**: runs the same per-particle Gaussian brightness fit as
  the "Brightness (Conf)" tool on the selected stacks, using the pixel size read
  directly from the file.
* **Overlay FOVs**: merge selected stacks on top of each other with per-channel
  colormaps, contrast, and opacity.

The heavy lifting for the readers/plotters is shared with the existing confocal
tools so behaviour stays consistent across PsyFit.
"""

import os
import io
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import streamlit as st

from utils import file_uploader_with_clear, plot_brightness, plot_histogram
from tools.confocal_brightness import integrate_dat
from tools.confocal_visualizer import (
    get_custom_lut,
    blend_images,
    add_scale_bar,
    get_norm,
)

# The OBF reader is an optional dependency; degrade gracefully if it is missing
# so the rest of PsyFit still loads.
try:
    from msr_reader import OBFFile, imspector_xml_to_dict
    _MSR_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - only hit when dep is absent
    OBFFile = None
    imspector_xml_to_dict = None
    _MSR_IMPORT_ERROR = _e


# Colormaps offered throughout the tool (custom LUTs first, then matplotlib).
LUT_OPTIONS = [
    "Greyscale", "Red hot", "Green hot", "Cyan hot", "Blue", "Green", "Pink",
    "hot", "magma", "viridis", "inferno", "plasma", "gray", "bone", "ocean",
]

# Substring patterns used to surface the "important" acquisition parameters out
# of the (large, deeply nested) Imspector metadata dictionary. Matching is done
# against the *leaf* key (last path segment) and is first-match-wins in this
# order, so e.g. "wavelength" is claimed by the wavelength category and does not
# leak into the power/window categories (whose tokens it happens to contain).
KEY_PARAM_PATTERNS = {
    "Excitation wavelength": ("wavelength", "exc_wl", "laser_wl", "excwl"),
    "Excitation intensity / power": ("power", "intensity"),
    "Imaging window / range": ("range", "fov", "roi", "psz", "pixel_size",
                               "field", "window", "offset"),
    "Dwell time": ("dwell", "pixel_time", "tdwell", "time_per"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _to_display_2d(arr):
    """Collapse an arbitrary-rank stack to a 2D float image for display/analysis.

    Anything above 2D (z-stacks, time series, extra channels) is reduced with a
    max-intensity projection over the leading axes, matching how the confocal
    tools treat multi-dimensional TIFFs.
    """
    arr = np.squeeze(np.asarray(arr))
    # Treat a trailing RGB(A) sample axis as colour channels -> luminance.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    while arr.ndim > 2:
        arr = arr.max(axis=0)
    if arr.ndim < 2:
        return None
    return arr.astype(float)


@st.cache_data(show_spinner=False)
def load_msr(file_bytes, filename):
    """Parse one ``.msr`` file into a list of per-stack dictionaries.

    Cached on the raw bytes so re-runs (colormap tweaks, selection changes) do
    not re-read the file. ``OBFFile`` needs a real path, so the upload is spilled
    to a temp file that is removed once parsing completes.
    """
    tmp_path = None
    stacks = []
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".msr", delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with OBFFile(tmp_path) as obf:
            for idx in range(obf.num_stacks):
                header = obf.stack_headers[idx]

                # Physical pixel size (metres -> nm); take the first spatial dim.
                try:
                    px_m = obf.pixel_size(idx)
                except Exception:
                    px_m = []
                pix_size_nm = float(px_m[0] * 1e9) if px_m else None

                # Shape / dimension labels straight from the file footer.
                try:
                    shape_info = obf.shapes[idx]
                    dim_labels = list(shape_info.dimension_names)
                    dim_sizes = list(shape_info.sizes)
                except Exception:
                    dim_labels, dim_sizes = [], []

                # Imspector XML metadata -> nested dict (may be absent on newer
                # Imspector versions, so guard every step).
                meta = {}
                try:
                    xml = obf.get_imspector_xml_metadata(idx)
                    if xml:
                        meta = imspector_xml_to_dict(xml) or {}
                except Exception:
                    meta = {}

                try:
                    raw = obf.read_stack(idx)
                    image = _to_display_2d(raw)
                    ndim = int(np.squeeze(np.asarray(raw)).ndim)
                except Exception:
                    image, ndim = None, None

                stacks.append({
                    "index": idx,
                    "name": header.name or f"Stack {idx}",
                    "image": image,
                    "ndim": ndim,
                    "pix_size_nm": pix_size_nm,
                    "dim_labels": dim_labels,
                    "dim_sizes": dim_sizes,
                    "metadata": meta,
                })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return stacks


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def flatten_metadata(obj, prefix=""):
    """Flatten a nested metadata dict/list into ``[(dotted.path, value_str)]``."""
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(flatten_metadata(v, path))
    elif isinstance(obj, (list, tuple)):
        # Short lists are rendered inline; long ones are indexed.
        if len(obj) <= 6 and all(not isinstance(x, (dict, list, tuple)) for x in obj):
            rows.append((prefix, ", ".join(str(x) for x in obj)))
        else:
            for i, v in enumerate(obj):
                rows.extend(flatten_metadata(v, f"{prefix}[{i}]"))
    else:
        rows.append((prefix, str(obj)))
    return rows


def key_parameters(flat_rows):
    """Pick out the acquisition parameters the user most cares about.

    Returns ``{category: [(path, value), ...]}`` for the categories in
    ``KEY_PARAM_PATTERNS`` whose dotted path contains a matching substring.
    """
    found = {cat: [] for cat in KEY_PARAM_PATTERNS}
    for path, value in flat_rows:
        # Match against the last two path segments: geometry values are often a
        # scalar under a named group (e.g. "scan.range.x"), so the leaf alone
        # ("x") is meaningless. Two segments capture the group name too.
        match_str = ".".join(path.lower().split(".")[-2:])
        for cat, patterns in KEY_PARAM_PATTERNS.items():
            if any(p in match_str for p in patterns):
                found[cat].append((path, value))
                break  # first-match-wins: no double counting across categories
    return {cat: rows for cat, rows in found.items() if rows}


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------
def _sel_key(filename, idx):
    return f"msr_sel::{filename}::{idx}"


def selected_indices(filename, stacks):
    """Return the indices the user has checked for the given file."""
    return [
        s["index"] for s in stacks
        if st.session_state.get(_sel_key(filename, s["index"]), False)
    ]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_preview(filename, stacks, cmap_name, log_scale, min_pct, max_pct):
    """Thumbnail grid of every stack with a select checkbox under each."""
    st.subheader("Preview & select images")
    st.caption(
        "Tick the images you want to carry into **Brightness** and **Overlay**. "
        "This is the working set drawn from this .msr."
    )

    c1, c2 = st.columns(2)
    if c1.button("Select all", key=f"selall_{filename}"):
        for s in stacks:
            st.session_state[_sel_key(filename, s["index"])] = True
        st.rerun()
    if c2.button("Clear selection", key=f"selnone_{filename}"):
        for s in stacks:
            st.session_state[_sel_key(filename, s["index"])] = False
        st.rerun()

    n_cols = 3
    lut = get_custom_lut(cmap_name)
    for row_start in range(0, len(stacks), n_cols):
        cols = st.columns(n_cols)
        for col, s in zip(cols, stacks[row_start:row_start + n_cols]):
            with col:
                img = s["image"]
                if img is None:
                    st.warning(f"Stack '{s['name']}' has no 2D image.")
                    continue

                fig, ax = plt.subplots(figsize=(3, 3))
                norm = get_norm(img, min_pct, max_pct, log_scale)
                ax.imshow(img, cmap=lut, norm=norm, origin="lower")
                add_scale_bar(ax, img.shape, s["pix_size_nm"])
                ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                px = f"{s['pix_size_nm']:.1f} nm/px" if s["pix_size_nm"] else "px size ?"
                dims = "×".join(str(d) for d in s["dim_sizes"]) or "—"
                st.caption(f"**{s['name']}**  \n{dims}  ·  {px}"
                           + ("  ·  multi-dim (projected)" if (s["ndim"] or 0) > 2 else ""))
                st.checkbox("Use this image", key=_sel_key(filename, s["index"]))


def render_metadata(filename, stacks):
    """Per-stack key parameters up top, full metadata behind a dropdown."""
    st.subheader("Saved metadata")

    names = [f"[{s['index']}] {s['name']}" for s in stacks]
    pick = st.selectbox("Stack", names, key=f"meta_pick_{filename}")
    s = stacks[names.index(pick)]

    # File-derived geometry is always available (independent of XML metadata).
    geo_cols = st.columns(3)
    geo_cols[0].metric("Pixel size",
                       f"{s['pix_size_nm']:.1f} nm" if s["pix_size_nm"] else "—")
    geo_cols[1].metric("Dimensions",
                       "×".join(str(d) for d in s["dim_sizes"]) or "—")
    geo_cols[2].metric("Axes", ", ".join(s["dim_labels"]) or "—")

    flat = flatten_metadata(s["metadata"])
    if not flat:
        st.info(
            "No Imspector XML metadata is stored in this stack. "
            "Newer Imspector versions may omit it; the file-derived geometry "
            "above is still exact."
        )
        return

    # Highlighted key acquisition parameters.
    key_params = key_parameters(flat)
    if key_params:
        st.markdown("**Key acquisition parameters**")
        for cat, rows in key_params.items():
            with st.container(border=True):
                st.markdown(f"*{cat}*")
                st.dataframe(
                    pd.DataFrame(rows, columns=["parameter", "value"]),
                    hide_index=True, use_container_width=True,
                )

    # Full metadata behind a collapsed dropdown, with a search filter.
    with st.expander("All saved metadata", expanded=False):
        query = st.text_input(
            "Filter", key=f"meta_filter_{filename}_{s['index']}",
            placeholder="e.g. wavelength, power, dwell",
        )
        df_meta = pd.DataFrame(flat, columns=["parameter", "value"])
        if query:
            mask = df_meta["parameter"].str.contains(query, case=False, na=False)
            df_meta = df_meta[mask]
        st.dataframe(df_meta, hide_index=True, use_container_width=True,
                     height=400)
        st.download_button(
            "Download metadata (CSV)",
            df_meta.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(filename)[0]}_stack{s['index']}_metadata.csv",
            mime="text/csv",
            key=f"meta_dl_{filename}_{s['index']}",
        )


def render_brightness(filename, stacks, sel, cmap_name, log_scale,
                      threshold_std, min_r2, dwell_us, line_acc, px_override_nm):
    """Run the confocal per-particle brightness fit on the selected stacks."""
    st.subheader("Confocal brightness")
    if not sel:
        st.info("Select at least one image in **Preview & select** first.")
        return

    st.markdown(r"$Brightness = \frac{Amplitude}{Dwell \times Accumulation}$")

    all_results = []
    processed = {}
    by_index = {s["index"]: s for s in stacks}

    for idx in sel:
        s = by_index[idx]
        img = s["image"]
        if img is None:
            continue

        pix_nm = px_override_nm if px_override_nm else (s["pix_size_nm"] or 100.0)
        pix_um = pix_nm / 1000.0
        dwell_s = dwell_us / 1e6

        df = integrate_dat(
            img, dwell_s, line_acc, f"{filename}::{s['name']}",
            threshold_std=threshold_std, pix_size_um=pix_um, min_fit_r2=min_r2,
        )
        if not df.empty:
            all_results.append(df)
        processed[s["name"]] = {"image": img, "df": df, "pix_nm": pix_nm}

    if not processed:
        st.warning("No usable 2D images among the selected stacks.")
        return

    col_viz, col_data = st.columns([2, 1])
    with col_viz:
        name = st.selectbox("View image", list(processed.keys()),
                            key=f"bright_view_{filename}")
        data = processed[name]
        df = data["df"]
        st.caption(f"{name} · {data['pix_nm']:.1f} nm/px · {len(df)} spots")
        fig = plot_brightness(
            data["image"], df, show_fits=True, normalization=log_scale,
            pix_size_um=data["pix_nm"] / 1000.0, cmap=_mpl_cmap(cmap_name),
            interactive=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_data:
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.metric("Total spots", len(combined))
            st.metric("Mean brightness",
                      f"{combined['brightness_integrated'].mean():.0f} pps")
            fig_h, _, _ = plot_histogram(
                combined,
                min_val=combined["brightness_integrated"].min(),
                max_val=combined["brightness_integrated"].max(),
                num_bins=30,
            )
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)
            st.download_button(
                "Download results (CSV)",
                combined.to_csv(index=False).encode("utf-8"),
                file_name=f"{os.path.splitext(filename)[0]}_brightness.csv",
                mime="text/csv", key=f"bright_dl_{filename}",
            )
        else:
            st.warning("No spots detected in the selected images.")


def _mpl_cmap(name):
    """Map our LUT names onto something ``plot_brightness`` understands."""
    plotly_ok = {"hot", "magma", "viridis", "inferno", "plasma", "gray"}
    return name if name in plotly_ok else "hot"


def _resize_to(img, shape):
    """Nearest-size resample of a 2D image onto ``shape`` (for overlay alignment)."""
    if img.shape == shape:
        return img
    zoom_factors = (shape[0] / img.shape[0], shape[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def render_overlay(filename, stacks, sel, log_scale):
    """Merge the selected stacks into a single blended RGB overlay."""
    st.subheader("Overlay fields of view")
    if len(sel) < 1:
        st.info("Select images in **Preview & select** to overlay them.")
        return

    by_index = {s["index"]: s for s in stacks}
    chosen = [by_index[i] for i in sel if by_index[i]["image"] is not None]
    if not chosen:
        st.warning("No usable 2D images among the selected stacks.")
        return

    # Per-channel controls.
    st.markdown("**Channel settings**")
    settings = {}
    for i, s in enumerate(chosen):
        cols = st.columns([2.2, 1.2, 1.0, 1.0, 1.2])
        cols[0].markdown(f"**{s['name']}**")
        lut = cols[1].selectbox(
            "LUT", LUT_OPTIONS, index=i % len(LUT_OPTIONS),
            key=f"ov_lut_{filename}_{s['index']}", label_visibility="collapsed")
        min_p = cols[2].number_input(
            "Min%", 0.0, 100.0, 1.0, key=f"ov_min_{filename}_{s['index']}",
            label_visibility="collapsed")
        max_p = cols[3].number_input(
            "Max%", 0.0, 100.0, 99.5, key=f"ov_max_{filename}_{s['index']}",
            label_visibility="collapsed")
        opacity = cols[4].slider(
            "Opacity", 0.0, 1.0, 1.0, key=f"ov_op_{filename}_{s['index']}",
            label_visibility="collapsed")
        settings[s["index"]] = {"lut": lut, "min_p": min_p, "max_p": max_p,
                                "opacity": opacity}

    align = st.checkbox(
        "Resample all channels to a common size", value=True,
        help="Overlay assumes the images share a field of view. Enable to "
             "resample differently-sized stacks onto a common pixel grid.")

    target_shape = max((s["image"].shape for s in chosen),
                       key=lambda sh: sh[0] * sh[1])

    blended = None
    ref_px = None
    for s in chosen:
        img = s["image"]
        if align:
            img = _resize_to(img, target_shape)
        cfg = settings[s["index"]]
        lut = get_custom_lut(cfg["lut"])
        norm = get_norm(img, cfg["min_p"], cfg["max_p"], log_scale)
        rgba = np.asarray(lut(norm(img)))
        rgba[..., 3] = cfg["opacity"]
        # Pre-multiply colour by opacity so additive blending respects it.
        rgba[..., :3] *= cfg["opacity"]
        if blended is None:
            blended = np.zeros(rgba.shape[:2] + (4,), dtype=float)
            ref_px = s["pix_size_nm"]
        if blended.shape[:2] != rgba.shape[:2]:
            st.warning(
                "Selected images have different sizes; enable "
                "'Resample all channels to a common size' to overlay them.")
            return
        blended = blend_images(blended, rgba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(blended, origin="lower")
    add_scale_bar(ax, blended.shape[:2],
                  ref_px if not align else (ref_px * (chosen[0]["image"].shape[0] / target_shape[0])
                                            if ref_px else None))
    ax.axis("off")
    ax.set_title(" + ".join(s["name"] for s in chosen), fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Offer a PNG export of the composite.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.download_button(
        "Download overlay (PNG)", buf.getvalue(),
        file_name=f"{os.path.splitext(filename)[0]}_overlay.png",
        mime="image/png", key=f"ov_dl_{filename}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    if OBFFile is None:
        st.error(
            "The `msr-reader` package is required to read .msr files but could "
            f"not be imported: {_MSR_IMPORT_ERROR}.\n\n"
            "Install it with `pip install msr-reader`."
        )
        return

    with st.sidebar:
        st.subheader("Files")
        uploaded = file_uploader_with_clear(
            "Upload .msr files", key="msr_uploads", type=["msr"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        st.subheader("Display")
        cmap_name = st.selectbox("Preview colormap", LUT_OPTIONS, index=7)  # 'hot'
        log_scale = st.toggle("Log scale", value=False)
        min_pct = st.slider("Min contrast %", 0.0, 100.0, 1.0)
        max_pct = st.slider("Max contrast %", 0.0, 100.0, 99.5)

        st.markdown("---")
        st.subheader("Brightness settings")
        threshold_std = st.slider("Detection threshold (σ above mean)", 1.0, 10.0, 5.0)
        min_r2 = st.slider("Min R²", 0.0, 1.0, 0.85)
        dwell_us = st.number_input("Dwell time (µs)", value=1000.0, min_value=0.1,
                                   step=10.0, format="%.1f")
        line_acc = st.number_input("Line accumulation", value=1, min_value=1, step=1)
        px_override_nm = st.number_input(
            "Pixel size override (nm, 0 = use file)", value=0.0, min_value=0.0,
            step=10.0, format="%.1f",
            help="Leave at 0 to use the pixel size read from each stack.")

    if not uploaded:
        st.info("Upload one or more .msr files to begin.")
        return

    file_names = [f.name for f in uploaded]
    active_name = st.selectbox("File", file_names)
    active_file = uploaded[file_names.index(active_name)]

    try:
        stacks = load_msr(active_file.getvalue(), active_name)
    except Exception as e:
        st.error(f"Failed to parse {active_name}: {type(e).__name__}: {e}")
        return

    if not stacks:
        st.warning("No stacks found in this .msr file.")
        return

    sel = selected_indices(active_name, stacks)
    st.caption(f"{len(stacks)} images in file · {len(sel)} selected")

    tab_prev, tab_meta, tab_bright, tab_overlay = st.tabs(
        ["Preview and select", "Metadata", "Brightness", "Overlay"]
    )
    with tab_prev:
        render_preview(active_name, stacks, cmap_name, log_scale, min_pct, max_pct)
    with tab_meta:
        render_metadata(active_name, stacks)
    with tab_bright:
        # Re-read selection: the preview tab may have just changed it.
        render_brightness(
            active_name, stacks, selected_indices(active_name, stacks),
            cmap_name, log_scale, threshold_std, min_r2, dwell_us, line_acc,
            px_override_nm,
        )
    with tab_overlay:
        render_overlay(active_name, stacks,
                       selected_indices(active_name, stacks), log_scale)


if __name__ == "__main__":
    run()
