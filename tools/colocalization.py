
# streamlit entry module for "UNDER CONSTRUCTION Colocalization Set"
# This script relies on helper functions defined in utils.py (must be importable on PYTHONPATH).
# Do NOT modify any functions in utils.py; this module only orchestrates UI + data flow.

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# Import Streamlit and provide a compatibility shim for older code that still calls experimental_rerun.
import streamlit as st
if not hasattr(st, "experimental_rerun"):
    if hasattr(st, "rerun"):
        st.experimental_rerun = st.rerun  # type: ignore

# Optionally quiet known non-fatal warnings originating from downstream plotting/fits.
import warnings
try:
    from scipy.optimize import OptimizeWarning  # type: ignore
except Exception:
    class OptimizeWarning(Warning): ...
warnings.filterwarnings("ignore", message="Adding colorbar to a different Figure")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=OptimizeWarning)

# Constant pixel size used throughout
PIX_SIZE_UM = 0.1

# Ensure we can import the coloc utils file the user provided
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils  # provided by the user

# Try to import sif_parser for preflight checks
try:
    import sif_parser
except Exception:
    sif_parser = None


def _full_image_cps_from_path(path: str):
    """Read the FULL SIF frame and convert to cps using metadata (no cropping)."""
    if sif_parser is None:
        raise RuntimeError("sif_parser is required to load SIF images from path")
    arr, meta = sif_parser.np_open(path, ignore_corrupt=True)
    # Expect shape (frames, H, W) or (H, W) depending on parser; take first frame
    a = np.asarray(arr)
    if a.ndim == 3:
        img_raw = a[0]
    else:
        img_raw = a
    gain = float(meta.get("GainDAC", 1.0)) or 1.0
    exp = float(meta.get("ExposureTime", 1.0)) or 1.0
    acc = float(meta.get("AccumulatedCycles", 1.0)) or 1.0
    cps = img_raw * (5.0 / gain) / exp / acc
    return cps, meta


_TMP_DIR = Path(tempfile.gettempdir()) / "coloc_sifs"
_TMP_DIR.mkdir(parents=True, exist_ok=True)

def _persist_to_temp(uploaded_file) -> Path:
    """Write UploadedFile to a temp file with original name (ensures .sif extension & .name).
    Returns the filesystem Path. We also reset the UploadedFile pointer for other tools.
    """
    data = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    # Use original filename; ensure uniqueness by namespacing under a per-session dir
    fname = Path(uploaded_file.name).name
    path = _TMP_DIR / fname
    # If file exists with different content, overwrite
    with open(path, "wb") as out:
        out.write(data)
    return path

def _preflight_path(path: Path) -> bool:
    if sif_parser is None:
        return True
    try:
        arr, meta = sif_parser.np_open(str(path), ignore_corrupt=True)
        arr = np.asarray(arr) if arr is not None else None
        return bool(arr is not None and arr.size > 0 and arr.ndim >= 2)
    except Exception:
        return False

def _load_fit_csvs(csv_files) -> Dict[str, pd.DataFrame]:
    mapping: Dict[str, pd.DataFrame] = {}
    if not csv_files:
        return mapping
    if len(csv_files) == 1:
        df = pd.read_csv(csv_files[0])
        name_col = None
        for c in df.columns:
            if c.lower() in ("sif_name", "filename", "file", "image", "sif"):
                name_col = c
                break
        if name_col is not None:
            for name, sub in df.groupby(name_col):
                base = os.path.basename(str(name))
                base_noext = os.path.splitext(base)[0]
                mapping[base_noext] = sub.reset_index(drop=True)
            return mapping
    for f in csv_files:
        base = os.path.basename(f.name if hasattr(f, "name") else str(f))
        base_noext = os.path.splitext(base)[0]
        try:
            mapping[base_noext] = pd.read_csv(f)
        except Exception as e:
            st.warning(f"Could not read CSV '{base}': {e}")
    return mapping


def _build_df_dict(sif_files, fit_map: Dict[str, pd.DataFrame], threshold: int, region: str, signal_guess_ucnp="976", signal_guess_dye="638") -> Tuple[Dict[str, dict], Dict[str, Tuple[pd.DataFrame, np.ndarray]], List[str], List[str]]:
    """Return df_dicts + two lists: skipped (errors) and soft_failed (preflight warnings).
    - df_dict_obj[name] = {"df": df, "image": img_full}
    - df_dict_tuple[name] = (df, img_full)
    - skipped: files that errored during integrate
    - soft_failed: files that failed preflight but still attempted processing
    """
    df_dict_obj: Dict[str, dict] = {}
    df_dict_tuple: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    skipped: List[str] = []
    soft_failed: List[str] = []

    # Persist all files to disk with original names
    path_map: Dict[str, Path] = {}
    for f in sif_files:
        try:
            path_map[f.name] = _persist_to_temp(f)
        except Exception as e:
            skipped.append(f.name)
            st.error(f"Failed to stage {f.name} to temp: {e}")

    for fname, path in path_map.items():
        # Soft preflight by PATH (do not block on failure)
        if not _preflight_path(path):
            soft_failed.append(fname)
        try:
            # Always build the FULL image from path
            img_full, meta = _full_image_cps_from_path(str(path))
        except Exception as e:
            skipped.append(fname)
            st.error(f"Failed to read full image for {fname}: {e}")
            continue

        # Try to obtain a DF from utils.integrate_sif (optional)
        df_est = None
        try:
            out = utils.integrate_sif(str(path), threshold=threshold, region=region, signal=("UCNP" if "976" in fname else ("Dye" if "638" in fname else "Unknown")), pix_size_um=PIX_SIZE_UM)
            if isinstance(out, tuple) and len(out) >= 3 and hasattr(out[2], "columns"):
                df_est = out[2]
        except Exception:
            pass  # Optional

        base_noext = os.path.splitext(os.path.basename(fname))[0]
        df = fit_map.get(base_noext, None) or df_est or pd.DataFrame()
        df_dict_obj[fname] = {"df": df, "image": img_full}
        df_dict_tuple[fname] = (df, img_full)

    return df_dict_obj, df_dict_tuple, skipped, soft_failed

def run():
    st.title("Colocalization Set (UNDER CONSTRUCTION)")
    st.caption("Single-file Streamlit app wrapper that uses your `utils.py` functions without modifying them.")

    
with st.sidebar:
    st.header("Inputs")
    sif_files = st.file_uploader("SIF files (UCNP + Dye)", type=["sif"], accept_multiple_files=True)
    csv_help = "Optional: upload one combined CSV with a 'sif_name'/'file' column, or per-image CSVs."
    fit_csvs = st.file_uploader("Fit CSVs (optional)", type=["csv"], accept_multiple_files=True, help=csv_help)

    st.divider()
    st.header("Fitting parameters")
    threshold = st.number_input("Threshold", min_value=0, value=2, help="Higher is more selective")
    region = st.selectbox("Region", options=["1","2","3","4","all"], index=4)

    st.header("Display")
    show_fits = st.checkbox("Overlay fits / circles / labels", value=True)
    cmap = st.selectbox("Colormap", options=["magma", "viridis", "plasma", "hot", "gray", "hsv"], index=0)
    univ_minmax = st.checkbox("Use universal min/max across all images", value=False)
    norm_choice = st.selectbox("Normalization", ["None", "LogNorm", "Normalize (0-99% percentile)"])
    save_format = st.selectbox("Save format", ["SVG", "PNG", "PDF"], index=0)

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    fit_map = _load_fit_csvs(fit_csvs)
    df_dict_obj, df_dict_tuple, skipped, soft_failed = _build_df_dict(sif_files, fit_map, threshold=threshold, region=region)
    if soft_failed:
        st.warning("Preflight could not verify these SIFs (still processed): " + ", ".join(soft_failed))
    if skipped:
        st.warning("Skipped file(s) due to processing errors: " + ", ".join(skipped))

    tab_grid, tab_pairs = st.tabs(["Grid of SIFs", "Paired UCNP ↔ Dye"])

    with tab_grid:
        st.subheader("All SIFs (grid)")
        normalization = None
        if norm_choice == "LogNorm":
            normalization = "log"
        elif norm_choice.startswith("Normalize"):
            normalization = "percentile"
        if hasattr(utils, "plot_all_sifs"):
            
try:
    # Try with full kwargs including cmap if supported
    results_df = utils.plot_all_sifs(
        sif_files=sif_files,
        df_dict=df_dict_obj,
        colocalization_radius=max(1, int(round(2 / PIX_SIZE_UM))),
        show_fits=show_fits,
        pix_size_um=PIX_SIZE_UM,
        normalization=normalization,
        save_format=save_format,
        univ_minmax=univ_minmax,
        cmap=cmap
    )
except TypeError:
    try:
        # Retry without cmap
        results_df = utils.plot_all_sifs(
            sif_files=sif_files,
            df_dict=df_dict_obj,
            colocalization_radius=max(1, int(round(2 / PIX_SIZE_UM))),
            show_fits=show_fits,
            pix_size_um=PIX_SIZE_UM,
            normalization=normalization,
            save_format=save_format,
            univ_minmax=univ_minmax,
        )
    except TypeError:
        # Minimal signature fallback
        results_df = utils.plot_all_sifs(sif_files, df_dict_obj)
        st.info("utils.plot_all_sifs signature differs; used minimal call.")

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                st.success(f"Found {len(results_df)} colocalized pairs across all images.")
                st.dataframe(results_df)
                st.download_button(
                    "Download colocalization results (CSV)",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="colocalization_results.csv",
                    mime="text/csv",
                )
        else:
            st.warning("utils.plot_all_sifs not found; skipping grid rendering.")

    with tab_pairs:
        st.subheader("UCNP ↔ Dye pairing & colocalization")
        if hasattr(utils, "sort_UCNP_dye_sifs") and hasattr(utils, "match_ucnp_dye_files") and hasattr(utils, "coloc_subplots"):
            ucnp_files, dye_files = utils.sort_UCNP_dye_sifs(sif_files)
            if len(ucnp_files) == 0 or len(dye_files) == 0:
                st.info("Need both UCNP and Dye files (filenames should contain '976' for UCNP and '638' for Dye by default).")
            else:
                pairs = utils.match_ucnp_dye_files(ucnp_files, dye_files)
                st.caption(f"Paired {len(pairs)} UCNP↔Dye sets.")
                collected = []
                for ucnp_f, dye_f in pairs:
                    try:
                        df_local = utils.coloc_subplots(
                            ucnp_file=ucnp_f,
                            dye_file=dye_f,
                            df_dict=df_dict_tuple,
                            colocalization_radius=max(1, int(round(coloc_radius_um / pix_size_um))),
                            show_fits=show_fits,
                            pix_size_um=pix_size_um
                        )
                        if isinstance(df_local, pd.DataFrame) and not df_local.empty:
                            df_local = df_local.copy()
                            df_local["ucnp_sif"] = ucnp_f.name
                            df_local["dye_sif"] = dye_f.name
                            collected.append(df_local)
                    except Exception as e:
                        st.error(f"coloc_subplots failed for {ucnp_f.name} vs {dye_f.name}: {e}")
                if collected:
                    out = pd.concat(collected, ignore_index=True)
                    st.success(f"Aggregated {len(out)} colocalized matches.")
                    st.dataframe(out)
                    st.download_button(
                        label="Download paired colocalization (CSV)",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="paired_colocalization.csv",
                        mime="text/csv",
                    )
        else:
            st.warning("One or more pairing/coloc helpers missing in utils.py; cannot run paired mode.")

if __name__ == "__main__":
    run()


# After pairing tab, show aggregate histograms/scatter like your single-sif tool
st.divider()
st.subheader("Aggregate analysis")
# Combine all per-file DFs (fits) for histogram
all_dfs = []
for k, v in df_dict_obj.items():
    dfk = v.get("df", pd.DataFrame())
    if isinstance(dfk, pd.DataFrame) and not dfk.empty and "brightness_fit" in dfk.columns:
        dfk = dfk.copy()
        dfk["source_sif"] = k
        all_dfs.append(dfk)
combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

if combined_df.empty:
    st.info("No fits available to plot histograms yet (upload fit CSVs or ensure integrate_sif returns fits).")
else:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Histogram controls
    bvals = combined_df["brightness_fit"].values
    default_min = float(np.min(bvals))
    default_max = float(np.max(bvals))
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        user_min = st.text_input("Min Brightness (pps)", value=f"{default_min:.2e}")
    with colB:
        user_max = st.text_input("Max Brightness (pps)", value=f"{default_max:.2e}")
    with colC:
        num_bins = st.number_input("# Bins", value=50)

    try:
        vmin = float(user_min)
        vmax = float(user_max)
    except Exception:
        st.warning("Enter valid numeric bounds for histogram.")
        vmin, vmax = default_min, default_max

    if vmin < vmax:
        fig_hist, axh = plt.subplots(figsize=(5,3))
        axh.hist(np.clip(bvals, vmin, vmax), bins=int(num_bins))
        axh.set_xlabel("Brightness (pps)")
        axh.set_ylabel("Count")
        st.pyplot(fig_hist)

# If a colocalization results_df exists (from grid or pairs), show simple distance scatter if column present
if 'results_df' in locals() and isinstance(results_df, pd.DataFrame) and not results_df.empty and "distance" in results_df.columns:
    import matplotlib.pyplot as plt
    fig_sc, ax = plt.subplots(figsize=(5,3))
    ax.scatter(results_df["distance"], results_df.get("num_dyes", pd.Series([1]*len(results_df))), alpha=0.6)
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Number of Dyes per PSF")
    st.pyplot(fig_sc)
