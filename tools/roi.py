"""Shared helpers for a user-drawn rectangular Region Of Interest (ROI).

The analysis tools (Brightness WF/Conf, Get Spectra, Monomer Estimation) crop
the image to a named region before detecting particles. This module adds a
"Custom" option: the user drags a rectangle on a reference frame and analysis is
restricted to that box, with the box coordinates written into exported CSVs.

An ROI is a 4-tuple of raw-image pixel indices ``(row0, row1, col0, col1)`` --
i.e. numpy slice bounds into the *analysis* array (the same non-flipped array
that ``integrate_sif`` / ``integrate_dat`` operate on). Callers can slice
directly with ``image[row0:row1, col0:col1]``.

Drawing uses Plotly's box-select (via ``st.plotly_chart(on_select=...)``) rather
than an image-canvas component, so it stays compatible with current Streamlit.
Plotly reports selection coordinates as image array indices, so the returned
``(row0, row1, col0, col1)`` index ``image_2d`` directly -- no flip arithmetic.
The ``flip_display`` flag only chooses the visual y-orientation (``origin``) so
the drawing view matches how each tool normally shows its frames.
"""

import os
import tempfile

import numpy as np
import streamlit as st
import plotly.express as px

try:
    import sif_parser
except Exception:  # pragma: no cover - only needed for the .sif reader
    sif_parser = None

# Named matplotlib colormaps -> Plotly continuous color scales.
_CMAP_TO_PLOTLY = {
    "gray": "gray", "grey": "gray", "hot": "hot", "viridis": "viridis",
    "plasma": "plasma", "magma": "magma", "inferno": "inferno", "hsv": "hsv",
}

# Column names stamped into exported dataframes.
ROI_COLUMNS = ("roi_row0", "roi_row1", "roi_col0", "roi_col1")


def read_sif_raw_from_path(path):
    """Return the raw (non-flipped) counts-per-second array for a .sif path.

    This is the same orientation ``integrate_sif`` crops, so an ROI drawn on it
    slices that array directly. Returns ``None`` on failure.
    """
    if sif_parser is None:
        return None
    image_data, metadata = sif_parser.np_open(path, ignore_corrupt=True)
    image_data = image_data[0]  # (H, W)
    gain = metadata.get("GainDAC", 1) or 1
    exposure = metadata["ExposureTime"]
    accumulate = metadata["AccumulatedCycles"]
    return image_data * (5.0 / gain) / exposure / accumulate


def read_first_sif_raw(uploaded_files):
    """Read the first uploaded .sif and return ``(name, raw_cps_image)``.

    Returns ``(None, None)`` if nothing could be read.
    """
    if not uploaded_files or sif_parser is None:
        return None, None

    temp_dir = os.path.join(tempfile.gettempdir(), "roi_temp")
    os.makedirs(temp_dir, exist_ok=True)

    for uf in uploaded_files:
        file_path = os.path.join(temp_dir, uf.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
            return uf.name, read_sif_raw_from_path(file_path)
        except Exception as e:  # noqa: BLE001 - surface but keep trying
            st.warning(f"Could not read {uf.name} for ROI drawing: {e}")

    return None, None


def _selection_box(state):
    """Pull the drawn rectangle out of a Plotly selection state, or None.

    Returns ``(x0, x1, y0, y1)`` in image data (index) coordinates.
    """
    sel = None
    if isinstance(state, dict):
        sel = state.get("selection")
    elif state is not None:
        sel = getattr(state, "selection", None)
    if not sel:
        return None
    box = sel.get("box") if isinstance(sel, dict) else getattr(sel, "box", None)
    if not box:
        return None

    b = box[-1]  # most recent rectangle
    xs = b.get("x") if isinstance(b, dict) else None
    ys = b.get("y") if isinstance(b, dict) else None
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def draw_roi(image_2d, key, *, cmap="gray", log=False, display_width=512,
             flip_display=True):
    """Render ``image_2d`` and let the user box-select a rectangular ROI on it.

    Uses Plotly's box-select tool. Returns ``(row0, row1, col0, col1)`` slice
    bounds into ``image_2d``, or ``None`` if no usable rectangle is selected.
    ``flip_display`` only sets the visual y-orientation (True -> ``origin='lower'``
    to match the tools' native display); the returned bounds index the array the
    same way regardless.
    """
    image_2d = np.asarray(image_2d, dtype=float)
    H, W = image_2d.shape

    disp = image_2d
    if log:
        disp = np.log1p(np.clip(image_2d - np.nanmin(image_2d), 0, None))

    origin = "lower" if flip_display else "upper"
    scale = _CMAP_TO_PLOTLY.get(str(cmap).lower(), "gray")
    fig = px.imshow(disp, origin=origin, aspect="equal",
                    color_continuous_scale=scale)
    fig.update_layout(
        dragmode="select",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        height=520,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    st.caption("Use the **Box Select** tool (top-right of the plot) to drag a "
               "rectangle. Draw again to replace it.")
    state = st.plotly_chart(
        fig, key=key, on_select="rerun", use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
    )

    box = _selection_box(state)
    if box is None:
        return None
    x0, x1, y0, y1 = box  # data coords == array indices (col, row)

    col0 = int(np.clip(np.floor(x0), 0, W))
    col1 = int(np.clip(np.ceil(x1), 0, W))
    row0 = int(np.clip(np.floor(y0), 0, H))
    row1 = int(np.clip(np.ceil(y1), 0, H))

    if row1 - row0 < 2 or col1 - col0 < 2:
        return None

    return (row0, row1, col0, col1)


def roi_columns(roi):
    """Return the ROI-coordinate columns as a dict for stamping onto a df."""
    if roi is None:
        return {c: np.nan for c in ROI_COLUMNS}
    row0, row1, col0, col1 = roi
    return {"roi_row0": row0, "roi_row1": row1, "roi_col0": col0, "roi_col1": col1}


def stamp_roi(df, roi):
    """Add ROI-coordinate columns to ``df`` in place and return it."""
    if df is None:
        return df
    for col, val in roi_columns(roi).items():
        df[col] = val
    return df
