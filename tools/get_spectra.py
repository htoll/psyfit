
import os
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import streamlit as st

import sys

import utils

PIX_SIZE_UM = 0.1  # fixed

try:
    from tools.process_files import process_files as _process_files_external  # type: ignore
except Exception:
    _process_files_external = None

# --- Helpers ---
def _process_files_fallback(uploaded_files, region, threshold=1, signal="UCNP", pix_size_um=PIX_SIZE_UM, sig_threshold=0.3):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
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


# --- App ---
def run():

    with st.sidebar:
        st.header("Inputs")
        sif_files = st.file_uploader("SIF files", type=["sif"], accept_multiple_files=True)

        st.divider()
        st.header("Fitting")
        threshold = st.number_input("Threshold", min_value=0, value=2)
        radius_px = st.number_input("Colocalization radius (pixels)", min_value=1, value=2)
        show_loc_fits = st.checkbox("Show localizations", value=False)
        show_spectra = st.checkbox("Show corresponding spectra", value=True)

        st.header("Display")
        cmap = st.selectbox("Colormap", options=["magma","viridis","plasma","hot","gray","hsv"], index=0)
        use_lognorm = st.checkbox("Log image scaling", value=True)
        show_colorbars = st.checkbox("Show colorbars on images", value=False)  # ← Add this


    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

#     # Localize
#     u_data, _ = _process_files(sif_files, region="Mr Beam", threshold=threshold, signal="UCNP")


#     # Prepare Matplotlib
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LogNorm

#     # Accumulate matched-peak rows
#     tab_pairs, tab_plots = st.tabs(["Pairs", "Plots"])

#     with tab_pairs:
#         matched_rows = []
#         overall_placeholder = st.empty()
#         overall_ucnp_hits = overall_ucnp_total = 0
#         overall_dye_hits  = overall_dye_total  = 0

#     # Render each pair as a row with 2 columns
#     for u_name, d_name in pairs:
#         u_bundle = u_data.get(u_name, {})
#         d_bundle = d_data.get(d_name, {})
#         u_df = u_bundle.get("df", pd.DataFrame())
#         d_df = d_bundle.get("df", pd.DataFrame())
#         # Prefer CSV fits if present
#         base_u = os.path.splitext(os.path.basename(u_name))[0]
#         base_d = os.path.splitext(os.path.basename(d_name))[0]
#         u_img = u_bundle.get("image", None)
#         d_img = d_bundle.get("image", None)

#         colL, colR = st.columns(2)
#         with colL:
#             st.markdown(f"**UCNP:** {u_name}")

#             if isinstance(u_img, np.ndarray):
#                 fig_u, ax_u = plt.subplots(figsize=(5,5))
#                 ax_u.set_xticks([]); ax_u.set_yticks([])
#                 norm = LogNorm() if use_lognorm else None
#                 im_u = ax_u.imshow(u_img + 1, cmap=cmap, norm=norm, origin="lower")  
#                 if show_colorbars:
#                     fig_u.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)          
#             else:
#                 fig_u, ax_u = plt.subplots(figsize=(5,5))
#                 ax_u.text(0.5,0.5,"No image", ha="center", va="center"); ax_u.axis("off")

#         with colR:
#             st.markdown(f"**Dye:** {d_name}")
#             if isinstance(d_img, np.ndarray):
#                 fig_d, ax_d = plt.subplots(figsize=(5,5))
#                 ax_d.set_xticks([]); ax_d.set_yticks([])
#                 norm = LogNorm() if use_lognorm else None
#                 im_d = ax_d.imshow(d_img + 1, cmap=cmap, norm=norm, origin="lower")
#                 if show_colorbars:
#                     fig_d.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
#             else:
#                 fig_d, ax_d = plt.subplots(figsize=(5,5))
#                 ax_d.text(0.5,0.5,"No image", ha="center", va="center"); ax_d.axis("off")

#         # Compute coloc mask & overlay
#         u_mask, d_mask, pair_idx = _compute_coloc_mask(u_df, d_df, radius_px=radius_px)
#         if isinstance(u_df, pd.DataFrame) and not u_df.empty and u_mask is not None:
#             u_total = len(u_df)
#             u_hits = int(u_mask.sum())
#             percent_ucnp_coloc = 100.0 * u_hits / max(u_total, 1)
#         else:
#             u_total = 0; u_hits = 0; percent_ucnp_coloc = 0.0

        
#         if isinstance(d_df, pd.DataFrame) and not d_df.empty and d_mask is not None:
#             d_total = len(d_df)
#             d_hits = int(d_mask.sum())
#             percent_dye_coloc = 100.0 * d_hits / max(d_total, 1)
#         else:
#             d_total = 0; d_hits = 0; percent_dye_coloc = 0.0
#         st.markdown(f"**Colocalized:** UCNP {u_hits}/{u_total} ({percent_ucnp_coloc:.1f}%) — Dye {d_hits}/{d_total} ({percent_dye_coloc:.1f}%)")

        
#         overall_ucnp_hits += u_hits
#         overall_ucnp_total += u_total
#         overall_dye_hits  += d_hits
#         overall_dye_total  += d_total



#         # Overlays
#         if show_all_fits:
#             _overlay_circles(ax_u, u_df, color="white", alpha=0.7, label=False)
#             _overlay_circles(ax_d, d_df, color="white", alpha=0.7, label=False)
#         if show_coloc_fits and u_mask is not None and d_mask is not None:
#             _overlay_circles(ax_u, u_df[u_mask], color="lime", alpha=0.9, label=False)
#             _overlay_circles(ax_d, d_df[d_mask], color="lime", alpha=0.9, label=False)

#         # Collect matched pairs data rows (if any)
#         if pair_idx:
#             for i_u, i_d, dist in pair_idx:
#                 row_u = u_df.loc[i_u] if i_u in u_df.index else None
#                 row_d = d_df.loc[i_d] if i_d in d_df.index else None
#                 if row_u is not None and row_d is not None:
#                     matched_rows.append({
#                         "ucnp_sif": u_name,
#                         "dye_sif": d_name,
#                         "ucnp_x_pix": row_u.get("x_pix", np.nan),
#                         "ucnp_y_pix": row_u.get("y_pix", np.nan),
#                         "ucnp_brightness": row_u.get("brightness_fit", np.nan),
#                         "dye_x_pix": row_d.get("x_pix", np.nan),
#                         "dye_y_pix": row_d.get("y_pix", np.nan),
#                         "dye_brightness": row_d.get("brightness_fit", np.nan),
#                         "distance_px": dist,
#                     })


#         with colL: st.pyplot(fig_u)
#         with colR: st.pyplot(fig_d)
#         overall_ucnp_pct = 100.0 * overall_ucnp_hits / max(overall_ucnp_total, 1)
#         overall_dye_pct  = 100.0 * overall_dye_hits  / max(overall_dye_total, 1)
#         overall_placeholder.markdown(
#             f"### Overall colocalized: "
#             f"UCNP {overall_ucnp_hits}/{overall_ucnp_total} ({overall_ucnp_pct:.1f}%) — "
#             f"Dye {overall_dye_hits}/{overall_dye_total} ({overall_dye_pct:.1f}%)"
#         )


#     # Download matched results CSV
#     if matched_rows:
#         matched_df = pd.DataFrame(matched_rows)
#         st.session_state['coloc_matched_df'] = matched_df

#         st.download_button(
#             "Download colocalized pairs (CSV)",
#             data=matched_df.to_csv(index=False).encode("utf-8"),
#             file_name="colocalized_pairs.csv",
#             mime="text/csv",
#         )




#         with tab_plots:
#             st.subheader("Plots")
#             matched_df = st.session_state.get("coloc_matched_df", pd.DataFrame())
#             if matched_df is None or matched_df.empty:
#                 st.info("No matched peaks yet — open the Pairs tab first.")
#             else:
#                 mode = st.radio("Single-particle brightness", ["Manual (enter pps)","Automatic (Gaussian μ)"], index=0, help="Automatic assumes most PSFs are single particles. If this is not true, use Manual.")
#                 st.warning("Automatic mode assumes the majority of PSFs are single particles.")

#                 if mode.startswith("Automatic"):
#                     try:
#                         from scipy.stats import norm
#                         uvals = matched_df["ucnp_brightness"].astype(float).to_numpy()
#                         dvals = matched_df["dye_brightness"].astype(float).to_numpy()
#                         import numpy as _np
#                         uvals = uvals[_np.isfinite(uvals) & (uvals > 0)]
#                         dvals = dvals[_np.isfinite(dvals) & (dvals > 0)]
#                         mu_ucnp, _ = norm.fit(uvals) if uvals.size else (_np.nan, _np.nan)
#                         mu_dye,  _ = norm.fit(dvals) if dvals.size else (_np.nan, _np.nan)
#                         single_ucnp_brightness = float(mu_ucnp) if _np.isfinite(mu_ucnp) else 1.0
#                         single_dye_brightness  = float(mu_dye)  if _np.isfinite(mu_dye)  else 1.0
#                         st.caption(f"Estimated single UCNP = {single_ucnp_brightness:.3e} pps, single Dye = {single_dye_brightness:.3e} pps")
#                     except Exception as e:
#                         st.error(f"Gaussian fit failed: {e}")
#                         single_ucnp_brightness = 1.0
#                         single_dye_brightness = 1.0
#                 else:
#                     c1, c2 = st.columns(2)
#                     with c1:
#                         single_ucnp_brightness = st.number_input("Single UCNP brightness (pps)", min_value=0.0, value=1e5, format="%.3e")
#                     with c2:
#                         single_dye_brightness  = st.number_input("Single Dye brightness (pps)", min_value=0.0, value=1e5, format="%.3e")

#                 md = matched_df.copy()
#                 md["num_ucnps"] = md["ucnp_brightness"].astype(float) / max(single_ucnp_brightness, 1e-12)
#                 md["num_dyes"]  = md["dye_brightness"].astype(float)  / max(single_dye_brightness,  1e-12)


#                 thresh_factor = st.number_input(
#                                 "UCNP quality cutoff (× single UCNP brightness)",
#                                 min_value=0.0, max_value=1.0, value=0.3, step=0.05,
#                                 help="Exclude points with UCNP brightness below factor × single-UCNP brightness."
#                             )
#                 thresholded_df = md[md["ucnp_brightness"] >= thresh_factor * single_ucnp_brightness].copy()

#                 import matplotlib.pyplot as plt
#                 fig_sc2, ax_sc2 = plt.subplots(figsize=(6,5))
#                 ax_sc2.scatter(md["num_ucnps"].to_numpy(), md["num_dyes"].to_numpy(), alpha=0.6)
#                 ax_sc2.set_xlabel("Number of UCNPs per PSF")
#                 ax_sc2.set_ylabel("Number of Dyes per PSF")
#                 ax_sc2.set_title("Matched UCNPs")
#                 ax_sc2.set_xlim(0, 5)
#                 colA, colB = st.columns(2)
#                 with colA:
#                     st.pyplot(fig_sc2)

#                 msk = (thresholded_df["num_ucnps"] >= 0) & (thresholded_df["num_ucnps"] <= 2)
#                 y_subset = thresholded_df.loc[msk, "num_dyes"].dropna().to_numpy()
#                 fig_h2, ax_h2 = plt.subplots(figsize=(6,5))
#                 if y_subset.size:
#                     import numpy as _np
#                     mean_val = float(_np.mean(y_subset))
#                     ax_h2.hist(y_subset, bins=15, edgecolor="black")
#                     ax_h2.set_title(f"Single UCNPs: Mean = {mean_val:.1f}")
#                 else:
#                     ax_h2.hist([], bins=15, edgecolor="black")
#                     ax_h2.set_title("Single UCNPs: no data in [0, 2] after threshold")
#                 ax_h2.set_xlabel("Number of Dyes per Single UCNP")
#                 ax_h2.set_ylabel("Count")
#                 with colB:
#                     st.pyplot(fig_h2)
#                 st.download_button(
#                     "Download thresholded results (CSV)",
#                     data=thresholded_df.to_csv(index=False).encode("utf-8"),
#                     file_name="thresholded_results.csv",
#                     mime="text/csv",
#                 )
if __name__ == "__main__":
    run()
