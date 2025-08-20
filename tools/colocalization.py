
# streamlit entry module for "UNDER CONSTRUCTION Colocalization Set"
# This script relies on helper functions defined in utils.py (must be importable on PYTHONPATH).
# Do NOT modify any functions in utils.py; this module only orchestrates UI + data flow.

import io
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Ensure we can import the coloc utils file the user provided
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils  # provided by the user

def _load_fit_csvs(csv_files) -> Dict[str, pd.DataFrame]:
    """Map basenames (without extension) of SIF files to DataFrames.
    Accepts either one combined CSV (with a column 'sif_name' or 'file') or per-image CSVs.
    Required columns for colocalization overlays: x_pix, y_pix, sigx_fit, sigy_fit, brightness_fit.
    """
    mapping = {}
    if not csv_files:
        return mapping

    # Heuristic 1: if exactly one CSV and it has a column indicating the sif name, split by it
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

    # Otherwise, treat each CSV as per-image table
    for f in csv_files:
        base = os.path.basename(f.name if hasattr(f, "name") else str(f))
        base_noext = os.path.splitext(base)[0]
        try:
            mapping[base_noext] = pd.read_csv(f)
        except Exception as e:
            st.warning(f"Could not read CSV '{base}': {e}")
    return mapping

def _build_df_dict(sif_files, fit_map: Dict[str, pd.DataFrame], pix_size_um: float, signal_guess_ucnp="976", signal_guess_dye="638") -> Tuple[Dict[str, dict], Dict[str, Tuple[pd.DataFrame, np.ndarray]]]:
    """Return two parallel dictionaries for different utils functions:
    - df_dict_obj[name] = {"df": df, "image": img}
    - df_dict_tuple[name] = (df, img)
    Where 'name' is the file's displayed name (UploadedFile.name).
    If no DF is supplied for a given SIF, we provide an empty DF so plots still render.
    """
    df_dict_obj = {}
    df_dict_tuple = {}

    for f in sif_files:
        # Integrate SIF to get image & metadata via user's utils
        try:
            # Pick signal label used just for utils.integrate_sif title-keeping. Guess from filename.
            fname = f.name
            signal = "UCNP" if signal_guess_ucnp in fname else ("Dye" if signal_guess_dye in fname else "Unknown")
            img, meta, df_est = None, None, None
            # utils.integrate_sif returns image cps and metadata; if a fitted DF is returned as well, accept it.
            try:
                # Some versions may return (image, metadata)
                out = utils.integrate_sif(f, signal=signal, pix_size_um=pix_size_um)
                if isinstance(out, tuple) and len(out) >= 2:
                    img, meta = out[0], out[1]
                    # Optional third: df-like?
                    if len(out) >= 3 and hasattr(out[2], "columns"):
                        df_est = out[2]
                else:
                    img = out  # fallback
            except TypeError:
                # Different signature; call with minimal args
                img, meta = utils.integrate_sif(f), {}

            # Choose DataFrame: user-provided > estimated > empty
            base_noext = os.path.splitext(os.path.basename(fname))[0]
            df = fit_map.get(base_noext, None) or df_est or pd.DataFrame()

            df_dict_obj[fname] = {"df": df, "image": img}
            df_dict_tuple[fname] = (df, img)
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")
    return df_dict_obj, df_dict_tuple

def run():
    st.title("Colocalization Set (UNDER CONSTRUCTION)")
    st.caption("Single-file Streamlit app wrapper that uses your \`utils.py\` functions without modifying them.")

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
    df_dict_obj, df_dict_tuple = _build_df_dict(sif_files, fit_map, pix_size_um=pix_size_um)

    # Tabs: Grid view and Paired UCNP↔Dye view
    tab_grid, tab_pairs = st.tabs(["Grid of SIFs", "Paired UCNP ↔ Dye"])

    with tab_grid:
        st.subheader("All SIFs (grid)")
        # Convert UI normalization option into utils-expected enum
        normalization = None
        if norm_choice == "LogNorm":
            normalization = "log"
        elif norm_choice.startswith("Normalize"):
            normalization = "percentile"  # utils will treat specially if implemented

        # Try to call the user's grid plotter if available
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
                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                    st.success(f"Found {len(results_df)} colocalized pairs across all images.")
                    st.dataframe(results_df)
                    st.download_button(
                        "Download colocalization results (CSV)",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="colocalization_results.csv",
                        mime="text/csv",
                    )
            except TypeError:
                st.warning("\`utils.plot_all_sifs\` signature differs; falling back to basic plotting only.")
            except Exception as e:
                st.error(f"plot_all_sifs failed: {e}")
        else:
            st.warning("utils.plot_all_sifs not found; skipping grid rendering.")

    with tab_pairs:
        st.subheader("UCNP ↔ Dye pairing & colocalization")
        # Separate files and pair them using provided utils
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
                            # include identifiers
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

# Allow running directly via `streamlit run colocalization.py`
if __name__ == "__main__":
    run()
