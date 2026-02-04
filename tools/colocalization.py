import os
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import hashlib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

# Ensure utils is accessible if needed
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils

PIX_SIZE_UM = 0.1  # fixed

# --- Caching and Processing Helpers ---

def _hash_file(uf) -> str:
    try:
        b = uf.getbuffer()
    except Exception:
        pos = uf.tell()
        b = uf.read()
        uf.seek(pos)
    return hashlib.md5(b).hexdigest()

def _build_proc_key(sif_files, region_ucnp, region_dye, threshold, ucnp_id, dye_id):
    names_hashes = tuple(sorted((f.name, _hash_file(f)) for f in (sif_files or [])))
    return (names_hashes, str(region_ucnp), str(region_dye), int(threshold), str(ucnp_id), str(dye_id))

def _extract_common_stem(uploaded_files):
    if not uploaded_files:
        return "results"
    name = Path(uploaded_files[0].name).stem
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# --- Processing Functions (Unchanged) ---

import warnings
try:
    from scipy.optimize import OptimizeWarning  # type: ignore
except Exception:
    class OptimizeWarning(Warning):
        pass
warnings.filterwarnings("ignore", message="Adding colorbar to a different Figure")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=OptimizeWarning)

try:
    from tools.process_files import process_files as _process_files_external  # type: ignore
except Exception:
    _process_files_external = None

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

# --- Plotting Helpers ---

def HWT_aesthetic():
    """Applies basic aesthetic settings to the current matplotlib axes."""
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

def _autoscale_xy(ax, x, y, pad=0.05):
    import numpy as _np
    x = _np.asarray(x); y = _np.asarray(y)
    m = _np.isfinite(x) & _np.isfinite(y)
    ax.set_xlim(0, 5)
    if not m.any():
        return
    x_in = (x >= 0) & (x <= 5) & m
    if not x_in.any(): x_in = m
    yv = y[x_in]
    if yv.size == 0: return
    ymin, ymax = float(_np.nanmin(yv)), float(_np.nanmax(yv))
    dy = ymax - ymin
    if not _np.isfinite(dy) or dy <= 0:
        dy = max(1.0, abs(ymax) if _np.isfinite(ymax) else 1.0)
    ax.set_ylim(ymin - dy * pad, ymax + dy * pad)

def _overlay_circles(ax, df: pd.DataFrame, color: str, alpha: float, label: bool = False):
    from matplotlib.patches import Circle
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    required = {"x_pix","y_pix","sigx_fit","sigy_fit","brightness_fit"}
    if not required.issubset(df.columns):
        return
    for _, row in df.iterrows():
        try:
            rad_px = 4 * float(max(row["sigx_fit"], row["sigy_fit"])) / PIX_SIZE_UM
        except Exception:
            rad_px = 6.0
        ax.add_patch(Circle((row["x_pix"], row["y_pix"]), radius=rad_px, color=color, fill=False, linewidth=1.2, alpha=alpha))
        if label:
            ax.text(row["x_pix"] + 8, row["y_pix"] + 8, f"{row['brightness_fit']/1000:.1f} kpps",
                    color=color, fontsize=8, ha="center", va="center")

# --- Matching Logic ---

def _split_ucnp_dye(files: List[Any], ucnp_id="976", dye_id="638") -> Tuple[List[Any], List[Any]]:
    u_tok = str(ucnp_id).lower().strip()
    d_tok = str(dye_id).lower().strip()
    ucnp, dye = [], []
    for f in files:
        name = f.name if hasattr(f, "name") else str(f)
        lname = name.lower()
        has_ucnp = u_tok in lname if u_tok else False
        has_dye  = d_tok in lname if d_tok else False
        if has_ucnp and not has_dye:
            ucnp.append(f)
        elif has_dye and not has_ucnp:
            dye.append(f)
        elif has_ucnp and has_dye:
            st.warning(f"Filename matches both tokens — skipping: {name}")
        else:
            st.warning(f"Filename matches neither token — skipping: {name}")
    return ucnp, dye

def _match_ucnp_dye_files(ucnps: List[Any], dyes: List[Any]) -> List[Tuple[Any, Any]]:
    def get_seq_index(file_obj) -> int:
        name = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        matches = re.findall(r'\d+', name)
        return int(matches[-1]) if matches else -1

    all_files = []
    for f in ucnps:
        all_files.append({'file': f, 'type': 'u', 'idx': get_seq_index(f)})
    for f in dyes:
        all_files.append({'file': f, 'type': 'd', 'idx': get_seq_index(f)})

    all_files.sort(key=lambda x: x['idx'])

    pairs: List[Tuple[Any, Any]] = []
    i = 0
    while i < len(all_files) - 1:
        current = all_files[i]
        next_f = all_files[i+1]
        idx_diff = abs(current['idx'] - next_f['idx'])
        
        if (current['type'] != next_f['type']) and (idx_diff <= 1):
            u_file = current['file'] if current['type'] == 'u' else next_f['file']
            d_file = next_f['file'] if next_f['type'] == 'd' else current['file']
            pairs.append((u_file, d_file))
            i += 2 
        else:
            i += 1
    return pairs

def _compute_coloc_mask(df_u: pd.DataFrame, df_d: pd.DataFrame, radius_px: int):
    if df_u is None or df_d is None or df_u.empty or df_d.empty:
        return None, None, []
    if not {"x_pix","y_pix"}.issubset(df_u.columns) or not {"x_pix","y_pix"}.issubset(df_d.columns):
        return None, None, []
    u_mask = np.zeros(len(df_u), dtype=bool)
    d_mask = np.zeros(len(df_d), dtype=bool)
    pairs = []
    used_d = set()
    for i_u, row_u in df_u.iterrows():
        du = df_d.loc[~df_d.index.isin(used_d)]
        if du.empty:
            continue
        dx = du["x_pix"].values - row_u["x_pix"]
        dy = du["y_pix"].values - row_u["y_pix"]
        dist = np.hypot(dx, dy)
        j = dist.argmin()
        if dist[j] <= radius_px:
            d_idx = du.index[j]
            u_mask[df_u.index.get_loc(i_u)] = True
            d_mask[df_d.index.get_loc(d_idx)] = True
            pairs.append((i_u, d_idx, float(dist[j])))
            used_d.add(d_idx)
    return u_mask, d_mask, pairs

# --- Main App ---

def run():
    st.set_page_config(layout="wide") # Helps with narrow screen issues

    with st.sidebar:
        st.header("Inputs")
        sif_files = st.file_uploader("SIF files (UCNP + Dye)", type=["sif"], accept_multiple_files=True)
        stem = _extract_common_stem(sif_files)

        st.header("IDs")
        ucnp_id = st.text_input("UCNP ID token", value="976", help="Substring used to identify UCNP files (matched in filename).")
        dye_id  = st.text_input("Dye ID token",  value="638", help="Substring used to identify Dye files (matched in filename).")

        st.divider()
        st.header("Fitting")
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region_ucnp = st.selectbox("Region (UCNP)", options=["1","2","3","4","all"], index=0)
        region_dye  = st.selectbox("Region (Dye)",  options=["1","2","3","4","all"], index=0)
        radius_px = st.number_input("Colocalization radius (pixels)", min_value=1, value=2)

        st.header("Overlays")
        show_all_fits = st.checkbox("Show all fits", value=False)
        show_coloc_fits = st.checkbox("Show colocalized fits", value=True)

        st.header("Display")
        cmap = st.selectbox("Colormap", options=["magma","viridis","plasma","hot","gray","hsv"], index=0)
        use_lognorm = st.checkbox("Log image scaling", value=True)
        show_colorbars = st.checkbox("Show colorbars on images", value=False)

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    # --- Processing ---
    proc_key = _build_proc_key(sif_files, region_ucnp, region_dye, threshold, ucnp_id, dye_id)
    if "proc_key" not in st.session_state or st.session_state["proc_key"] != proc_key:
        ucnp_files, dye_files = _split_ucnp_dye(sif_files, ucnp_id=ucnp_id, dye_id=dye_id)
        u_data, _ = _process_files(ucnp_files, region=region_ucnp, threshold=threshold, signal="UCNP") if ucnp_files else ({}, pd.DataFrame())
        d_data, _ = _process_files(dye_files,  region=region_dye,  threshold=threshold, signal="dye")  if dye_files  else ({}, pd.DataFrame())

        u_names = sorted(list(u_data.keys()), key=natural_sort_key)
        d_names = sorted(list(d_data.keys()), key=natural_sort_key)
        pairs = _match_ucnp_dye_files(u_names, d_names)

        st.session_state.update({
            "proc_key": proc_key,
            "u_data": u_data,
            "d_data": d_data,
            "pairs": pairs,
        })
    else:
        u_data = st.session_state["u_data"]
        d_data = st.session_state["d_data"]
        pairs  = st.session_state["pairs"]

    st.caption(f"Matched {len(pairs)} UCNP↔Dye pairs.")

    # --- Pre-calculate Statistics to avoid layout jumping ---
    overall_ucnp_hits = overall_ucnp_total = 0
    overall_dye_hits  = overall_dye_total  = 0
    
    ucnp_pct_list = []
    dye_pct_list = []
    
    # We iterate once purely for stats (computation is fast, plotting is slow)
    for u_name, d_name in pairs:
        u_df = u_data.get(u_name, {}).get("df", pd.DataFrame())
        d_df = d_data.get(d_name, {}).get("df", pd.DataFrame())
        
        u_mask, d_mask, _ = _compute_coloc_mask(u_df, d_df, radius_px=radius_px)

        u_h = int(u_mask.sum()) if (u_mask is not None) else 0
        u_t = len(u_df) if isinstance(u_df, pd.DataFrame) else 0
        
        d_h = int(d_mask.sum()) if (d_mask is not None) else 0
        d_t = len(d_df) if isinstance(d_df, pd.DataFrame) else 0
        
        overall_ucnp_hits += u_h
        overall_ucnp_total += u_t
        overall_dye_hits += d_h
        overall_dye_total += d_t

        if u_t > 0: ucnp_pct_list.append(100.0 * u_h / u_t)
        if d_t > 0: dye_pct_list.append(100.0 * d_h / d_t)

    # Calculate overall aggregate %
    ov_u_pct = 100.0 * overall_ucnp_hits / max(overall_ucnp_total, 1)
    ov_d_pct = 100.0 * overall_dye_hits / max(overall_dye_total, 1)
    
    # Calculate SD across images
    sd_u = np.std(ucnp_pct_list) if ucnp_pct_list else 0.0
    sd_d = np.std(dye_pct_list) if dye_pct_list else 0.0

    # Display Overall Stats at the TOP
    st.markdown(
        f"### Overall Colocalized\n"
        f"**UCNP:** {overall_ucnp_hits}/{overall_ucnp_total} "
        f"({ov_u_pct:.1f}% ± {sd_u:.1f}%) "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Dye:** {overall_dye_hits}/{overall_dye_total} "
        f"({ov_d_pct:.1f}% ± {sd_d:.1f}%)"
    )
    
    st.divider()

    tab_pairs = st.container()
    tab_plots = st.container()

    matched_rows = []

    with tab_pairs:
        # --- Iterate Pairs for Display ---
        for u_name, d_name in pairs:
            u_bundle = u_data.get(u_name, {})
            d_bundle = d_data.get(d_name, {})
            u_df = u_bundle.get("df", pd.DataFrame())
            d_df = d_bundle.get("df", pd.DataFrame())
            u_img = u_bundle.get("image", None)
            d_img = d_bundle.get("image", None)

            # Re-compute mask for this specific pair to get display data
            u_mask, d_mask, pair_idx = _compute_coloc_mask(u_df, d_df, radius_px=radius_px)

            # Collect matched data rows
            if pair_idx:
                for i_u, i_d, dist in pair_idx:
                    row_u = u_df.loc[i_u] if i_u in u_df.index else None
                    row_d = d_df.loc[i_d] if i_d in d_df.index else None
                    if row_u is not None and row_d is not None:
                        matched_rows.append({
                            "ucnp_sif": u_name,
                            "dye_sif": d_name,
                            "ucnp_x_pix": row_u.get("x_pix", np.nan),
                            "ucnp_y_pix": row_u.get("y_pix", np.nan),
                            "ucnp_brightness": row_u.get("brightness_fit", np.nan),
                            "dye_x_pix": row_d.get("x_pix", np.nan),
                            "dye_y_pix": row_d.get("y_pix", np.nan),
                            "dye_brightness": row_d.get("brightness_fit", np.nan),
                            "distance_px": dist,
                        })

            # Stats for this specific image
            u_h = int(u_mask.sum()) if (u_mask is not None) else 0
            u_t = len(u_df) if isinstance(u_df, pd.DataFrame) else 0
            d_h = int(d_mask.sum()) if (d_mask is not None) else 0
            d_t = len(d_df) if isinstance(d_df, pd.DataFrame) else 0
            
            p_u = 100.0 * u_h / max(u_t, 1)
            p_d = 100.0 * d_h / max(d_t, 1)

            # 2. Setup Columns (UCNP | Dye | Recon)
            colL, colM, colR = st.columns(3)
            text_style = "min-height: 50px; line-height: 1.2; word-wrap: break-word; margin-bottom: 5px;"

            # -- Left: UCNP Image --
            with colL:
                st.markdown(
                f"<div style='{text_style}'><b>UCNP:</b><br>{u_name}</div>", 
                unsafe_allow_html=True
                            )
                if isinstance(u_img, np.ndarray):
                    fig_u, ax_u = plt.subplots(figsize=(5,5))
                    ax_u.set_xticks([]); ax_u.set_yticks([])
                    norm = LogNorm() if use_lognorm else None
                    im_u = ax_u.imshow(u_img + 1, cmap=cmap, norm=norm, origin="lower")
                    if show_colorbars:
                        fig_u.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
                else:
                    fig_u, ax_u = plt.subplots(figsize=(5,5))
                    ax_u.text(0.5,0.5,"No image", ha="center", va="center"); ax_u.axis("off")

            # -- Middle: Dye Image --
            with colM:
                st.markdown(
                f"<div style='{text_style}'><b>Dye:</b><br>{d_name}</div>", 
                unsafe_allow_html=True
                            )
                if isinstance(d_img, np.ndarray):
                    fig_d, ax_d = plt.subplots(figsize=(5,5))
                    ax_d.set_xticks([]); ax_d.set_yticks([])
                    norm = LogNorm() if use_lognorm else None
                    im_d = ax_d.imshow(d_img + 1, cmap=cmap, norm=norm, origin="lower")
                    if show_colorbars:
                        fig_d.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
                else:
                    fig_d, ax_d = plt.subplots(figsize=(5,5))
                    ax_d.text(0.5,0.5,"No image", ha="center", va="center"); ax_d.axis("off")

            # -- Right: Reconstruction Plot --
            with colR:
                st.markdown(
                f"<div style='{text_style}'><b>Reconstruction</b></div>", 
                unsafe_allow_html=True
                            )
                fig_r, ax_r = plt.subplots(figsize=(5,5))
                
                # Set bounds
                if isinstance(u_img, np.ndarray):
                    h, w = u_img.shape
                    ax_r.set_xlim(0, w); ax_r.set_ylim(0, h)
                elif isinstance(d_img, np.ndarray):
                    h, w = d_img.shape
                    ax_r.set_xlim(0, w); ax_r.set_ylim(0, h)
                else:
                    ax_r.axis("equal")
                
                # Plot UCNP (Grey)
                if not u_df.empty:
                    ax_r.scatter(u_df["x_pix"], u_df["y_pix"], c="dodgerblue", s=25, alpha=0.65, label="UCNP", edgecolors='none')


                # Plot Dye (Red)
                if not d_df.empty:
                    # Alpha set to 0.8 as requested
                    ax_r.scatter(d_df["x_pix"], d_df["y_pix"], c="firebrick", s=25, alpha=0.65, label="Dye", edgecolors='none')
                    # Add 'x' for coloc
                    if d_mask is not None and d_mask.any():
                        coloc_d = d_df[d_mask]
                        ax_r.scatter(coloc_d["x_pix"], coloc_d["y_pix"], marker="o", facecolors='none',  edgecolors='black', s=150, linewidths=1)

                ax_r.set_aspect('equal')
                ax_r.set_xticks([]); ax_r.set_yticks([])
                ax_r.legend(loc='upper right', fontsize='x-small', framealpha=0.8)
                st.pyplot(fig_r)

            # Apply Overlays to Images
            if show_all_fits:
                _overlay_circles(ax_u, u_df, color="white", alpha=0.7, label=False)
                _overlay_circles(ax_d, d_df, color="white", alpha=0.7, label=False)
            if show_coloc_fits and u_mask is not None and d_mask is not None:
                _overlay_circles(ax_u, u_df[u_mask], color="lime", alpha=0.9, label=False)
                _overlay_circles(ax_d, d_df[d_mask], color="lime", alpha=0.9, label=False)

            # Display Stats text
            st.markdown(f"**Colocalized:** UCNP {u_h}/{u_t} ({p_u:.1f}%) — Dye {d_h}/{d_t} ({p_d:.1f}%)")

            # Render the image plots
            with colL: st.pyplot(fig_u)
            with colM: st.pyplot(fig_d)
            
            st.divider()

    # --- CSV Download & Analysis Plots ---
    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)
        st.session_state['coloc_matched_df'] = matched_df

    with tab_plots:
        matched_df = st.session_state.get("coloc_matched_df", pd.DataFrame())
        if matched_df is None or matched_df.empty:
            st.info("No matched peaks yet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                single_ucnp_brightness = st.number_input("Single UCNP brightness (pps)", min_value=0.0, value=1e5, format="%.2e")
            with c2:
                single_dye_brightness  = st.number_input("Single Dye brightness (pps)", min_value=0.0, value=5e2, format="%.2e")

            md = matched_df.copy()
            md["num_ucnps"] = md["ucnp_brightness"].astype(float) / max(single_ucnp_brightness, 1e-12)
            md["num_dyes"]  = md["dye_brightness"].astype(float)  / max(single_dye_brightness,  1e-12)
            
            # CSV Download Logic
            meta = io.StringIO()
            meta.write("Single Emitter,Brightness(pps)\n")
            meta.write(f"single_ucnp_brightness_pps,{single_ucnp_brightness}\n")
            meta.write(f"single_dye_brightness_pps,{single_dye_brightness}\n")
            meta.write("\n")
            md_csv = io.StringIO()
            md.to_csv(md_csv, index=False)
            md_csv.seek(0)
            payload = (meta.getvalue() + md_csv.getvalue()).encode("utf-8")
            
            st.download_button(
                "Download results (CSV)",
                data=payload,
                file_name=f"{stem}_colocalized_results.csv",
                mime="text/csv",
            )

            thresh_factor = st.number_input(
                "UCNP quality cutoff (× single UCNP brightness)",
                min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Exclude points with UCNP brightness below factor × single-UCNP brightness."
            )
            thresholded_df = md[md["ucnp_brightness"] >= thresh_factor * single_ucnp_brightness].copy()

            fig_sc2, ax_sc2 = plt.subplots(figsize=(6,5))
            xvals = md["num_ucnps"].to_numpy(dtype=float)
            yvals = md["num_dyes"].to_numpy(dtype=float)
            ax_sc2.scatter(md["num_ucnps"].to_numpy(), md["num_dyes"].to_numpy(), alpha=0.6, clip_on=False)
            ax_sc2.set_xlabel("Number of UCNPs per PSF")
            ax_sc2.set_ylabel("Number of Dyes per PSF")
            ax_sc2.set_title("Matched UCNPs")
            _autoscale_xy(ax_sc2, xvals, yvals)
            HWT_aesthetic()
            ax_sc2.grid(True, which="both", linestyle=":", color="lightgrey", alpha=0.7)

            colA, colB = st.columns(2)
            with colA:
                st.pyplot(fig_sc2)

            msk = (thresholded_df["num_ucnps"] >= 0) & (thresholded_df["num_ucnps"] <= 2)
            y_subset = thresholded_df.loc[msk, "num_dyes"].dropna().to_numpy()
            fig_h2, ax_h2 = plt.subplots(figsize=(6,5))
            if y_subset.size:
                mean_val = float(np.mean(y_subset))
                ax_h2.hist(y_subset, bins=15, edgecolor="black")
                ax_h2.set_title(f"Single UCNPs: Mean = {mean_val:.1f}")
            else:
                ax_h2.hist([], bins=15, edgecolor="black")
                ax_h2.set_title("Single UCNPs: no data in [0, 2] after threshold")
            ax_h2.set_xlabel("Number of Dyes per Single UCNP")
            ax_h2.set_ylabel("Count")
            ax_h2.xaxis.set_major_locator(MaxNLocator(integer=True))
            HWT_aesthetic()
            if y_subset.size <= 2:
                ax_h2.set_ylim(0, 5)

            with colB:
                st.pyplot(fig_h2)

if __name__ == "__main__":
    run()
