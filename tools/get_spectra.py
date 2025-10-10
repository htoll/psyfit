
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

import Methods

import sif_parser

import pickle as pkl
import textwrap

PIX_SIZE_UM = 0.107 # from Mr Beam

# try:
#     from tools.process_files import process_files as _process_files_external  # type: ignore
# except Exception:
#     _process_files_external = None

# --- Helpers ---
def _process_files(uploaded_files, region="Mr Beam", threshold=1, signal="UCNP", pix_size_um=0.107, sig_threshold=0.3):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
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

def just_read_in(uploaded_files):
    full_frames = {}
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            image_data, metadata = sif_parser.np_open(str(file_path), ignore_corrupt=True)
            image_data = image_data[0]  # (H, W)

            gainDAC = metadata['GainDAC']
            if gainDAC == 0:
                gainDAC =1 #account for gain turned off
            exposure_time = metadata['ExposureTime']
            accumulate_cycles = metadata['AccumulatedCycles']

            # Normalize counts → photons
            image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles
            image_data_cps = np.flipud(image_data_cps)

            full_frames[uf.name] = image_data_cps
        except Exception as e:
            st.error(f"Error reading in {uf.name}: {e}")

    return full_frames

# def read_in_calibration(date, folder = "G:/Shared drives/SamPengLab/Alev_Studenikina/Multicolor/Heterogeneity/Calibration data/"):
#     import pickle as pkl

#     with open(f"{folder}saving_info_{date}_no_tracking.pkl", "rb") as f:
#         calibration = pkl.load(f)

#     with open(f"{folder}{date}_fits.pkl", "rb") as f:
#         calib_fits = pkl.load(f)

#     return calibration, calib_fits

def read_in_calibration(uploaded_files):
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())

        with open(file_path, "rb") as f:
            file = pkl.load(f)
        
        files.append(file)

    return files
    
def get_spectrum(coord, frame, calibration, calib_fits, background="na"):
    object, image = Methods.get_image(coord, np.array([val[1] for val in calibration.values()]), np.array([val[3] for val in calibration.values()]))

    xs, ys = image
    if round(ys) < 510:
        rows_to_average_over = [-2, -1, 0, 1, 2]
    elif round(ys) == 510:
        rows_to_average_over = [-2, -1, 0, 1]
    elif round(ys) == 511:
        rows_to_average_over = [-2, -1, 0]
    elif round(ys) > 511:
        raise ValueError("y coordinate too close to edge of image to extract spectrum")

    intensity = np.zeros(100)

    for j in rows_to_average_over:
        intensity += np.interp(np.linspace(xs - 90, xs+9, 100), np.arange(512), frame[round(ys)+j, :])
    intensity /= len(rows_to_average_over)

    # if rightmost side cropped
    if len(intensity) < 100:
        intensity = np.hstack((intensity, np.ones(100 - len(intensity))*intensity[-1]))

    k=0
    distances = [Methods.distance(coord, val[1]) for val in calibration.values()]
    sorted_ind = np.argsort(distances)
    ids_sorted_by_distance = np.array(list(calibration.keys()))[sorted_ind]
    closest = ids_sorted_by_distance[0]
    while closest not in calib_fits:
        k += 1
        closest = ids_sorted_by_distance[k]

    custom_pixel_to_wvl = Methods.exp(np.arange(100), *calib_fits[closest])

    nms = Methods.exp(np.arange(101)-0.5, *calib_fits[closest])
    nms_per_pixel = [nms[i+1]-nms[i] for i in range(100)]

    # if type(background) != str:
    #     background_subtract = np.interp(np.linspace(xs - 90, xs+9, 100), np.arange(512), background[round(ys)+j, :])
    #     if len(background_subtract) < 100:
    #         background_subtract = np.hstack((background_subtract, np.ones(100 - len(background_subtract))*background_subtract[-1]))
        
    #     return custom_pixel_to_wvl, intensity-background_subtract
    
    # else:
    return custom_pixel_to_wvl, intensity, nms_per_pixel
    

def fit_template_linear(y, template, return_params=False):
    """
    Fit y ~ a*template + b using least squares.
    Returns fitted background (a*template + b) and (a,b) if requested.
    """
    y = np.asarray(y).ravel()
    t = np.asarray(template).ravel()
    if y.shape != t.shape:
        raise ValueError("y and template must have same shape")

    # design matrix [template, 1]
    A = np.vstack([t, np.ones_like(t)]).T
    # solve least squares
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    bg = a * t + b
    if return_params:
        return bg, (a, b)
    return bg

def fit_template_sigma_clip(y, template, niter=10, sigma_thresh=1):
    y = np.asarray(y).ravel()
    t = np.asarray(template).ravel()
    mask = np.ones_like(y, dtype=bool)
    a = b = 0.0
    for _ in range(niter):
        if mask.sum() < 3:
            break
        bg, (a, b) = fit_template_linear(y[mask], t[mask], return_params=True)
        # build full bg for residuals
        full_bg = a*t + b
        resid = y - full_bg
        std = resid[mask].std(ddof=1) if mask.sum() > 1 else resid.std()
        # keep points that are not strong positive outliers (peaks)
        new_mask = resid < sigma_thresh * std
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask
    bg = a*t + b
    return bg, (a, b), mask



# --- App ---
def run():
    col1, col2 = st.columns([1, 1])
    # with col1:
    with st.sidebar:
        st.header("Inputs v0.1")
        sif_files = st.file_uploader("SIF files", type=["sif"], accept_multiple_files=True)

        background = st.file_uploader("Blank (optional)", type=["sif"], help='''
                    Upload image of an empty FOV under the same imaging conditions 
                    for background substraction. If absent, linear background estimation is performed.
        ''')
        # calibration_date = st.text_input("Calibration date", help="Please enter the date in the format like 250920")

        calibration = st.file_uploader("Calibration file", accept_multiple_files=False,type=["pkl"], help = '''
                                       Something of the form saving_info_YYMMDD.pkl
                                       or saving_info_YYMMDD_no_tracking.pkl [lighter version].
                                       Can be found in 
                                       G:\Shared drives\SamPengLab\Alev_Studenikina\Multicolor\Heterogeneity\Calibration data 
                                       ''')
        
        calib_fits = st.file_uploader("Calibration fits", accept_multiple_files=False, type=["pkl"],
 help = '''
                                       Something of the form YYMMDD_fits.pkl
                                       or YYMMDD_fits_456_474_548_667_803.pkl
                                       is more accurate actually.
                                       Can be found in 
                                       G:\Shared drives\SamPengLab\Alev_Studenikina\Multicolor\Heterogeneity\Calibration data 
                                       ''')

        st.divider()
        st.header("Fitting")
        threshold = st.number_input("Threshold", min_value=0, value=2)
        radius_px = st.number_input("Radius (pixels)", min_value=1, value=2)

        st.header("Display")
        # cmap = st.selectbox("Colormap", options=["gray","magma","viridis","plasma","hot","hsv"], index=0)
        # use_lognorm = st.checkbox("Log image scaling", value=True)
        vmax = st.number_input("vmax", value = 1000)
        show_colorbar = st.checkbox("Show colorbar", value=False)  

        st.header("Spectra")
        remove_overlapping = st.checkbox("Remove overlapping spectra", value=False)

        normalize = st.checkbox("Normalize spectra", value=False)

        no_dim = st.number_input("Exclude particles below a certain brightness threshold", value=0.0)

        scale_per_nm = st.checkbox("Scale intensity based on pixel bin size\n(can mess with background)", value=False)

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True

        if not sif_files:
            st.info("Upload SIF files to begin.")
            return
        
        if not calibration or not calib_fits:
            st.info("Please provide the calibration data you wish to use.")
            return
        

    # Prepare Matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    with col1:


        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.session_state.analyze_clicked:
            # Localize
            u_data, _ = _process_files(sif_files, region="Mr Beam", threshold=threshold, signal="UCNP")
            # u_data is a dictionary structured as {file_name: [dataframe with file info, image]}
            ####################################################################################
                    #     results.append({
                    #     'x_pix': center_x_refined,
                    #     'y_pix': center_y_refined,
                    #     'x_um': x0_fit,
                    #     'y_um': y0_fit,
                    #     'amp_fit': amp_fit,
                    #     'sigx_fit': sigx_fit,
                    #     'sigy_fit': sigy_fit,
                    #     'brightness_fit': brightness_fit,
                    #     'brightness_integrated': brightness_integrated
                    # })
            ####################################################################################

            full_images = just_read_in(sif_files)

            calibration, calib_fits = read_in_calibration([calibration, calib_fits])

            if background is not None:
                # Write uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                    tmp.write(background.getbuffer())
                    tmp_path = tmp.name
                blank, metadata = sif_parser.np_open(tmp_path)
                blank = blank[0]  # (H, W)

                gainDAC = metadata['GainDAC']
                if gainDAC == 0:
                    gainDAC =1 #account for gain turned off
                exposure_time = metadata['ExposureTime']
                accumulate_cycles = metadata['AccumulatedCycles']

                # Normalize counts → photons
                blank_cps = blank * (5.0 / gainDAC) / exposure_time / accumulate_cycles
                blank_cps = np.flipud(blank_cps)

                if blank_cps.shape[0] == 1:
                    background_image = blank_cps[0]
                else:
                    background_image = np.zeros((512, 512))
                    for img in blank_cps:
                        background_image += img
                    background_image /= blank_cps.shape[0]


            # st.markdown(f"**UCNP:**")
            for key, val in u_data.items():
                u_img = val["image"]
                # u_img = np.flip(u_img, axis=0)
                full_frame = full_images[key]

                fig, ax = plt.subplots(figsize=(5,5))

                im_u = ax.imshow(full_frame, cmap="gray", vmax=vmax, origin="upper")  
                ax.axis('off')
                if show_colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='10%', pad=0.2)
                    fig.colorbar(im_u, cax=cax)    

                ax.set_title("\n".join(textwrap.wrap(key, width=25)))

                df = val["df"]
                coords = {}
                for i, (_, row) in enumerate(df.iterrows()):
                    x = row["x_um"]/PIX_SIZE_UM
                    y = row["y_um"]/PIX_SIZE_UM

                    coords[i] = [x, 280-y]

                if remove_overlapping:
                    to_remove = []

                    for i in coords:
                        [x1, y1], _ = Methods.get_image(coords[i], [val[1] for val in calibration.values()], [val[3] for val in calibration.values()])
                        for j in range(i+1, len(coords)):
                            [x2, y2], _ = Methods.get_image(coords[j], [val[1] for val in calibration.values()], [val[3] for val in calibration.values()])
                            if np.abs(x1-x2) < 95 and np.abs(y1-y2) < 5:
                                to_remove.append(i)
                                to_remove.append(j)

                    for id in to_remove:
                        if id in coords:
                            del coords[id]

                too_dim = [i for i, (_, row) in enumerate(df.iterrows()) if row["brightness_fit"] < no_dim]

                for id in too_dim:
                    if id in coords:
                        del coords[id]

                colors = cm.rainbow(np.linspace(0, 1, len(coords.keys())))
                for i, id in enumerate(coords):
                    x, y = coords[id]
                    ax.scatter(x, y, s=2, c=colors[i])
                    # if label:
                    #     ax.text(row["x_pix"] + 8, row["y_pix"] + 8, f"{row['brightness_fit']/1000:.1f} kpps",
                    #             color=color, fontsize=8, ha="center", va="center")

                    # wvl, spec = get_spectrum(np.array([x, y]), full_frame, calibration, calib_fits)

                # Show the figure
                st.pyplot(fig)

        with col2:
                for key, val in u_data.items():
                    u_img = val["image"]
                    # u_img = np.flip(u_img, axis=0)
                    full_frame = full_images[key]

                    fig, ax = plt.subplots(figsize=(5,4))

                    # im_u = ax.imshow(full_frame, cmap="gray", origin="lower")  
                    # ax.axis('off')
                    # if show_colorbar:
                    #     divider = make_axes_locatable(ax)
                    #     cax = divider.append_axes('right', size='10%', pad=0.2)
                    #     fig.colorbar(im_u, cax=cax, location='left')    

                    # show localizations
                    df = val["df"]
                    # axs[0].scatter(df['x_um'], df['y_um'])
                    coords = {}
                    for i, (_, row) in enumerate(df.iterrows()):
                        x = row["x_um"]/PIX_SIZE_UM
                        y = row["y_um"]/PIX_SIZE_UM

                        coords[i] = [x, 280-y]

                    if remove_overlapping:
                        to_remove = []

                        for i in coords:
                            [x1, y1], _ = Methods.get_image(coords[i], [val[1] for val in calibration.values()], [val[3] for val in calibration.values()])
                            for j in range(i+1, len(coords)):
                                [x2, y2], _ = Methods.get_image(coords[j], [val[1] for val in calibration.values()], [val[3] for val in calibration.values()])
                                if np.abs(x1-x2) < 95 and np.abs(y1-y2) < 5:
                                    to_remove.append(i)
                                    to_remove.append(j)

                        for id in to_remove:
                            if id in coords:
                                del coords[id]


                    too_dim = [i for i, (_, row) in enumerate(df.iterrows()) if row["brightness_fit"] < no_dim]

                    for id in too_dim:
                        if id in coords:
                            del coords[id]
                    
                    if not scale_per_nm:
                        ax.set_title("Emission intensity per pixel")
                    
                    else:
                        ax.set_title("Emission intensity per nm")

                    colors = cm.rainbow(np.linspace(0, 1, len(coords.keys())))
                    for i, id in enumerate(coords):
                        x, y = coords[id]

                        # ax.scatter(x, y, color='r', s=0.1)
                        # if label:
                        #     ax.text(row["x_pix"] + 8, row["y_pix"] + 8, f"{row['brightness_fit']/1000:.1f} kpps",
                        #             color=color, fontsize=8, ha="center", va="center")
                        try:
                            wvl, spec, nms_per_pixel = get_spectrum(np.array([x, y]), full_frame, calibration, calib_fits)
                            
                            if background is not None:
                                #ax.plot(wvl, spec, c=colors[i])
                                bkg_wvl, bkg_spec, _ = get_spectrum(np.array([x,y]), background_image, calibration, calib_fits)
                                #ax.plot(wvl, bkg_spec, 'k--')
                                res = fit_template_sigma_clip(spec, bkg_spec, 10, 1)
                                bkg_fitted = res[0]
                                #ax.plot(wvl, bkg_fitted, 'k--') 
                                intensity_sans_bkg = spec-bkg_fitted

                                if not scale_per_nm:
                                    if not normalize:
                                        ax.plot(wvl, intensity_sans_bkg, c=colors[i])
                                    else:
                                        ax.plot(wvl, Methods.normalize(intensity_sans_bkg), c=colors[i])

                                else:
                                    intensity_scaled = intensity_sans_bkg/nms_per_pixel
                                    if not normalize:
                                        ax.plot(wvl, intensity_scaled, c=colors[i])
                                    else:
                                        ax.plot(wvl, Methods.normalize(intensity_scaled), c=colors[i])
                            
                            else:
                                intensity_sans_bkg = spec-np.min(spec)
                                if not scale_per_nm:
                                    if not normalize:
                                        ax.plot(wvl, intensity_sans_bkg, c=colors[i])
                                    else:
                                        ax.plot(wvl, Methods.normalize(intensity_sans_bkg), c=colors[i])

                                else:
                                    intensity_scaled = intensity_sans_bkg/nms_per_pixel
                                    if not normalize:
                                        ax.plot(wvl, intensity_scaled, c=colors[i])
                                    else:
                                        ax.plot(wvl, Methods.normalize(intensity_scaled), c=colors[i])
                        except Exception as e:
                            st.error(f"Error extracting spectrum for particle {id}")
                            continue
                    
                    ax.set_xlabel("Wavelength (nm)")

                    # Show the figure
                    st.pyplot(fig)
                            
   
                  
if __name__ == "__main__":
    run()
