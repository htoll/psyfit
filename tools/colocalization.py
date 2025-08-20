
# streamlit entry module for "UNDER CONSTRUCTION Colocalization Set"
# This script relies on helper functions defined in utils.py (must be importable on PYTHONPATH).
# Do NOT modify any functions in utils.py; this module only orchestrates UI + data flow.

import io
import os
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

def _bytesio_with_name(uploaded_file) -> io.BytesIO:
    """Return a fresh BytesIO carrying a .name attribute to mimic a real file object."""
    data = uploaded_file.read()
    # Reset original pointer so other parts of the app can reuse the UploadedFile
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    bio = io.BytesIO(data)
    try:
        bio.name = uploaded_file.name  # some libs (or your utils) inspect .name
    except Exception:
        pass
    return bio

def _preflight_nonblocking(bio_named: io.BytesIO) -> bool:
    """Soft check: returns True if readable; but failures do NOT block downstream processing."""
    if sif_parser is None:
        return True
    try:
        # Use a new handle for the read since np_open may consume it
        temp = io.BytesIO(bio_named.getvalue())
        try:
            temp.name = getattr(bio_named, "name", None) or "uploaded.sif"
        except Exception:
            pass
        arr, meta = sif_parser.np_open(temp, ignore_corrupt=True)
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

def _build_df_dict(sif_files, fit_map: Dict[str, pd.DataFrame], pix_size_um: float, signal_guess_ucnp="976", signal_guess_dye="638") -> Tuple[Dict[str, dict], Dict[str, Tuple[pd.DataFrame, np.ndarray]], List[str], List[str]]:
    """Return df_dicts + two lists: skipped (errors) and soft_failed (preflight warnings).
    - df_dict_obj[name] = {"df": df, "image": img}
    - df_dict_tuple[name] = (df, img)
    - skipped: files that errored during integrate
    - soft_failed: files that failed preflight but still attempted processing
    """
    df_dict_obj: Dict[str, dict] = {}
    df_dict_tuple: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    skipped: List[str] = []
    soft_failed: List[str] = []

    for f in sif_files:
        # Create a Named BytesIO that preserves .name
        bio = _bytesio_with_name(f)
        # Soft preflight: warn but do not skip
        if not _preflight_nonblocking(bio):
            soft_failed.append(f.name)
        try:
            fname = f.name
            signal = "UCNP" if signal_guess_ucnp in fname else ("Dye" if signal_guess_dye in fname else "Unknown")
            img, meta, df_est = None, None, None
            # Always pass a fresh stream to the utils call
            bio_call = io.BytesIO(bio.getvalue())
            try:
                bio_call.name = fname
            except Exception:
                pass
            try:
                out = utils.integrate_sif(bio_call, signal=signal, pix_size_um=pix_size_um)
                if isinstance(out, tuple) and len(out) >= 2:
                    img, meta = out[0], out[1]
                    if len(out) >= 3 and hasattr(out[2], "columns"):
                        df_est = out[2]
                else:
                    img = out
            except TypeError:
                bio_call2 = io.BytesIO(bio.getvalue())
                try:
                    bio_call2.name = fname
                except Exception:
                    pass
                out = utils.integrate_sif(bio_call2)
                if isinstance(out, tuple):
                    img = out[0]
                else:
                    img = out
                meta = {}

            base_noext = os.path.splitext(os.path.basename(fname))[0]
            df = fit_map.get(base_noext, None) or df_est or pd.DataFrame()
            df_dict_obj[fname] = {"df": df, "image": img}
            df_dict_tuple[fname] = (df, img)
        except Exception as e:
            skipped.append(f.name)
            st.error(f"Failed to process {f.name}: {e}")
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
        st.header("Parameters")
        pix_size_um = st.number_input("Pixel size (µm)", min_value=0.01, max_value=5.0, value=0.10, step=0.01, format="%.2f")
        coloc_radius_um = st.number_input("Colocalization radius (µm)", min_value=0.05, max_value=10.0, value=0.20, step=0.05, format="%.2f")
        show_fits = st.checkbox("Overlay fits / circles / labels (requires fit columns)", value=True)
        st.caption("Required DF columns: x_pix, y_pix, sigx_fit, sigy_fit, brightness_fit")

        st.subheader("Display options")
        univ_minmax = st.checkbox("Use universal min/max across all images", value=False)
        norm_choice = st.selectbox("Normalization", ["None", "LogNorm", "Normalize (0-99% percentile)"])
        save_format = st.selectbox("Save format", ["SVG", "PNG", "PDF"], index=0)

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    # Prepare data
    fit_map = _load_fit_csvs(fit_csvs)
    df_dict_obj, df_dict_tuple, skipped, soft_failed = _build_df_dict(sif_files, fit_map, pix_size_um=pix_size_um)
    if soft_failed:
        st.warning("Preflight could not verify these SIFs, but processing was attempted anyway: " + ", ".join(soft_failed))
    if skipped:
        st.warning("Skipped file(s) due to processing errors: " + ", ".join(skipped))

    # Tabs: Grid view and Paired UCNP↔Dye view
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
                results_df = utils.plot_all_sifs(
                    sif_files=sif_files,
                    df_dict=df_dict_obj,
                    colocalization_radius=max(1, int(round(coloc_radius_um / pix_size_um))),
                    show_fits=show_fits,
                    pix_size_um=pix_size_um,
                    normalization=normalization,
                    save_format=save_format,
                    univ_minmax=univ_minmax,
                )
            except TypeError:
                try:
                    results_df = utils.plot_all_sifs(sif_files, df_dict_obj)
                    st.info("utils.plot_all_sifs signature differs; used minimal call.")
                except Exception as e:
                    results_df = None
                    st.error(f"plot_all_sifs failed (minimal): {e}")
            except Exception as e:
                results_df = None
                st.error(f"plot_all_sifs failed: {e}")

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
