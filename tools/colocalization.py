
# streamlit entry module for "UNDER CONSTRUCTION Colocalization Set"
# This script wires your existing utils without modifying them.

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

import streamlit as st
if not hasattr(st, "experimental_rerun"):
    if hasattr(st, "rerun"):
        st.experimental_rerun = st.rerun  # type: ignore

import warnings
try:
    from scipy.optimize import OptimizeWarning  # type: ignore
except Exception:
    class OptimizeWarning(Warning):
        pass
warnings.filterwarnings("ignore", message="Adding colorbar to a different Figure")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=OptimizeWarning)

PIX_SIZE_UM = 0.1  # fixed

import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils  # provided by you

# Try to import your working pipeline
try:
    from tools.process_files import process_files as _process_files_external  # type: ignore
except Exception:
    _process_files_external = None

# Local fallback that mirrors your process_files implementation
def _process_files_fallback(uploaded_files, region, threshold=1, signal="UCNP", pix_size_um=PIX_SIZE_UM, sig_threshold=0.3):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "coloc_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            df, image_data_cps = utils.integrate_sif(
                str(file_path),
                region=region,
                threshold=threshold,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
            )
            processed_data[uf.name] = {"df": df, "image": image_data_cps}
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uf.name}: {e}")
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return processed_data, combined_df

def _process_files(uploaded_files, region, threshold, signal):
    if _process_files_external is not None:
        return _process_files_external(uploaded_files, region, threshold=threshold, signal=signal, pix_size_um=PIX_SIZE_UM)
    return _process_files_fallback(uploaded_files, region, threshold=threshold, signal=signal, pix_size_um=PIX_SIZE_UM)

def _split_ucnp_dye(files):
    # Use utils sorter if present
    if hasattr(utils, "sort_UCNP_dye_sifs"):
        return utils.sort_UCNP_dye_sifs(files)
    # Fallback by filename
    ucnp, dye = [], []
    for f in files:
        name = f.name.lower()
        if "976" in name and "638" not in name:
            ucnp.append(f)
        elif "638" in name and "976" not in name:
            dye.append(f)
    return ucnp, dye

def run():
    st.title("Colocalization Set (UNDER CONSTRUCTION)")
    st.caption("Streamlit wrapper that uses your `utils.py` and (if available) `tools.process_files`.")

    with st.sidebar:
        st.header("Inputs")
        sif_files = st.file_uploader("SIF files (UCNP + Dye)", type=["sif"], accept_multiple_files=True)
        csv_help = "Optional: upload one combined CSV with a 'sif_name'/'file' column, or per-image CSVs."
        fit_csvs = st.file_uploader("Fit CSVs (optional)", type=["csv"], accept_multiple_files=True, help=csv_help)

        st.divider()
        st.header("Fitting parameters")
        threshold = st.number_input("Threshold", min_value=0, value=2, help="Higher is more selective")
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], index=4)

        st.header("Display")
        show_fits = st.checkbox("Overlay fits / circles / labels", value=True)
        cmap = st.selectbox("Colormap", options=["magma", "viridis", "plasma", "hot", "gray", "hsv"], index=0)
        norm_choice = st.selectbox("Normalization", ["None", "LogNorm", "Normalize (0-99% percentile)"])
        univ_minmax = st.checkbox("Use universal min/max across all images", value=False)
        save_format = st.selectbox("Save format", ["SVG", "PNG", "PDF"], index=0)
        radius_px = st.number_input("Colocalization radius (pixels)", min_value=1, value=2)

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    # Optional: load fit CSVs (per-image or one combined with a sif_name/file column)
    fit_map: Dict[str, pd.DataFrame] = {}
    if fit_csvs:
        # combined
        if len(fit_csvs) == 1:
            df = pd.read_csv(fit_csvs[0])
            name_col = next((c for c in df.columns if c.lower() in ("sif_name","filename","file","image","sif")), None)
            if name_col:
                for name, sub in df.groupby(name_col):
                    base = os.path.basename(str(name))
                    fit_map[os.path.splitext(base)[0]] = sub.reset_index(drop=True)
        # per-image
        if not fit_map:
            for f in fit_csvs:
                base_noext = os.path.splitext(os.path.basename(f.name))[0]
                try:
                    fit_map[base_noext] = pd.read_csv(f)
                except Exception as e:
                    st.warning(f"Could not read CSV '{f.name}': {e}")

    # Split files by signal
    ucnp_files, dye_files = _split_ucnp_dye(sif_files)

    # Process each group with the correct detection mode
    ucnp_data, ucnp_combined = _process_files(ucnp_files, region=region, threshold=threshold, signal="UCNP") if ucnp_files else ({}, pd.DataFrame())
    dye_data, dye_combined   = _process_files(dye_files,  region=region, threshold=threshold, signal="dye")   if dye_files  else ({}, pd.DataFrame())

    # Merge into df_dicts that utils.coloc_subplots expects
    df_dict_obj: Dict[str, dict] = {}
    df_dict_tuple: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    for name, bundle in {**ucnp_data, **dye_data}.items():
        df = bundle.get("df", pd.DataFrame())
        img = bundle.get("image", None)
        # If a fit CSV exists for this base name, prefer it
        base_noext = os.path.splitext(os.path.basename(name))[0]
        if base_noext in fit_map:
            df = fit_map[base_noext]
        df_dict_obj[name] = {"df": df, "image": img}
        df_dict_tuple[name] = (df, img)

    # --- Tab 1: Grid of SIFs (full-frame from selected region) ---
    tab_grid, tab_pairs = st.tabs(["Grid of SIFs", "Paired UCNP ↔ Dye"])
    with tab_grid:
        st.subheader("All SIFs (grid)")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        images = [(n, v.get("image", None)) for n, v in df_dict_obj.items()]
        if not images:
            st.info("No images to display yet.")
        else:
            n_files = len(images)
            n_cols = 4
            n_rows = (n_files + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            # compute global min/max if requested
            vmin = vmax = None
            if univ_minmax:
                vals = [img for _, img in images if isinstance(img, np.ndarray)]
                if vals:
                    vmin = min(float(np.nanmin(v)) for v in vals if v is not None)
                    vmax = max(float(np.nanmax(v)) for v in vals if v is not None)

            norm = LogNorm() if norm_choice == "LogNorm" else None
            for ax, (name, img) in zip(axes, images):
                ax.set_xticks([]); ax.set_yticks([])
                if isinstance(img, np.ndarray):
                    ax.imshow(img + 1, cmap=cmap, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
                    ax.set_title(name, fontsize=10)
                else:
                    ax.text(0.5,0.5,"No image", ha="center", va="center")
                    ax.set_title(name, fontsize=10)
            for ax in axes[len(images):]:
                ax.axis("off")
            st.pyplot(fig)

    # --- Tab 2: Pairing & colocalization ---
    with tab_pairs:
        st.subheader("UCNP ↔ Dye pairing & colocalization")
        if hasattr(utils, "match_ucnp_dye_files") and hasattr(utils, "coloc_subplots"):
            pairs = utils.match_ucnp_dye_files(ucnp_files, dye_files) if ucnp_files and dye_files else []
            st.caption(f"Paired {len(pairs)} UCNP↔Dye sets.")

            collected = []
            required = {"x_pix","y_pix","sigx_fit","sigy_fit","brightness_fit"}
            for ucnp_f, dye_f in pairs:
                u_df = df_dict_tuple.get(ucnp_f.name, (pd.DataFrame(),None))[0]
                d_df = df_dict_tuple.get(dye_f.name, (pd.DataFrame(),None))[0]
                has_ucnp = isinstance(u_df, pd.DataFrame) and required.issubset(u_df.columns)
                has_dye  = isinstance(d_df, pd.DataFrame) and required.issubset(d_df.columns)
                show_fits_eff = show_fits and has_ucnp and has_dye

                try:
                    df_local = utils.coloc_subplots(
                        ucnp_file=ucnp_f,
                        dye_file=dye_f,
                        df_dict=df_dict_tuple,
                        colocalization_radius=radius_px,
                        show_fits=show_fits_eff,
                        pix_size_um=PIX_SIZE_UM,
                    )
                    if isinstance(df_local, pd.DataFrame) and not df_local.empty:
                        tmp = df_local.copy()
                        tmp["ucnp_sif"] = ucnp_f.name
                        tmp["dye_sif"]  = dye_f.name
                        collected.append(tmp)
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
            st.warning("Pairing utilities not found in utils.py")

    # --- Aggregate analysis (brightness histogram + optional distance scatter) ---
    st.divider()
    st.subheader("Aggregate analysis")
    all_dfs = []
    for bundle in {**ucnp_data, **dye_data}.values():
        df = bundle.get("df", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty and "brightness_fit" in df.columns:
            all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if combined_df.empty:
        st.info("No fits available to plot histograms yet (upload fit CSVs or ensure integrate_sif returns fits).")
    else:
        import matplotlib.pyplot as plt
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
            vmin = float(user_min); vmax = float(user_max)
        except Exception:
            st.warning("Enter valid numeric bounds for histogram.")
            vmin, vmax = default_min, default_max
        if vmin < vmax:
            fig_hist, axh = plt.subplots(figsize=(5,3))
            axh.hist(np.clip(bvals, vmin, vmax), bins=int(num_bins))
            axh.set_xlabel("Brightness (pps)"); axh.set_ylabel("Count")
            st.pyplot(fig_hist)

if __name__ == "__main__":
    run()
