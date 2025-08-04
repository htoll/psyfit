import sif_parser
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.feature import blob_log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import seaborn as sns

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from datetime import date

import streamlit as st
import io

def HWT_aesthetic():
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5,
                        "axes.labelsize": 14,
                        "axes.titlesize": 16})
    sns.set_palette("tab20c")  #pref'd are colorblind , tab20c , muted6 ,
    sns.despine()

def integrate_sif(sif, threshold=1, region='all', signal='UCNP', pix_size_um = 0.1, sig_threshold = 0.3):
    image_data, metadata = sif_parser.np_open(sif)
    image_data = image_data[0]  # (H, W)

    gainDAC = metadata['GainDAC']
    exposure_time = metadata['ExposureTime']
    accumulate_cycles = metadata['AccumulatedCycles']

    # Normalize counts → photons
    image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles

    radius_um_fine = 0.3
    radius_pix_fine = int(radius_um_fine / pix_size_um)

    # --- Crop image if region specified ---
    region = str(region)
    if region == '3':
        image_data_cps = image_data_cps[0:256, 0:256]
    elif region == '4':
        image_data_cps = image_data_cps[0:256, 256:512]
    elif region == '1':
        image_data_cps = image_data_cps[256:512, 0:256]
    elif region == '2':
        image_data_cps = image_data_cps[256:512, 256:512]
    elif region == 'custom': #accounting for misaligned 638 beam on 250610
        image_data_cps = image_data_cps[312:512, 56:256]

    # else → 'all': use full image

    # --- Detect peaks ---
    smoothed_image = gaussian_filter(image_data_cps, sigma=1)
    threshold_abs = np.mean(smoothed_image) + threshold * np.std(smoothed_image)

    if signal == 'UCNP':
        coords = peak_local_max(smoothed_image, min_distance=5, threshold_abs=threshold_abs)
    else:
        blobs = blob_log(smoothed_image, min_sigma=1, max_sigma=3, num_sigma=5, threshold=5 * threshold)
        coords = blobs[:, :2]

    #print(f"{os.path.basename(sif)}: Found {len(coords)} peaks in region {region}")

    results = []
    for center_y, center_x in coords:
        # Extract subregion
        sub_img, x0_idx, y0_idx = extract_subregion(image_data_cps, center_x, center_y, radius_pix_fine)

        # Refine peak
        blurred = gaussian_filter(sub_img, sigma=1)
        local_peak = peak_local_max(blurred, num_peaks=1)
        if local_peak.shape[0] == 0:
            continue
        local_y, local_x = local_peak[0]
        center_x_refined = x0_idx + local_x
        center_y_refined = y0_idx + local_y

        # Extract finer subregion
        sub_img_fine, x0_idx_fine, y0_idx_fine = extract_subregion(
            image_data_cps, center_x_refined, center_y_refined, radius_pix_fine
        )
        # Interpolate to 20x20 grid (like MATLAB)
        interp_size = 20
        zoom_factor = interp_size / sub_img_fine.shape[0]
        sub_img_interp = zoom(sub_img_fine, zoom_factor, order=1)  # bilinear interpolation

        # Prepare grid
        # y_indices, x_indices = np.indices(sub_img_fine.shape)
        # x_coords = (x_indices + x0_idx_fine) * pix_size_um
        # y_coords = (y_indices + y0_idx_fine) * pix_size_um
        interp_shape = sub_img_interp.shape
        y_indices, x_indices = np.indices(interp_shape)
        x_coords = (x_indices / interp_shape[1] * sub_img_fine.shape[1] + x0_idx_fine) * pix_size_um
        y_coords = (y_indices / interp_shape[0] * sub_img_fine.shape[0] + y0_idx_fine) * pix_size_um

        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        z_flat = sub_img_interp.ravel() #∆ variable name 250604

        # Initial guess
        amp_guess = np.max(sub_img_fine)
        offset_guess = np.min(sub_img_fine)
        x0_guess = center_x_refined * pix_size_um
        y0_guess = center_y_refined * pix_size_um
        sigma_guess = 0.15
        p0 = [amp_guess, x0_guess, sigma_guess, y0_guess, sigma_guess, offset_guess]

        # Fit
        try:
            #popt, _ = curve_fit(gaussian2d, (x_flat, y_flat), z_flat, p0=p0)
            def residuals(params, x, y, z):
                A, x0, sx, y0, sy, offset = params
                model = A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2))) + offset
                return model - z

            lb = [1, x0_guess - 1, 0.0, y0_guess - 1, 0.0, offset_guess * 0.5]
            ub = [2 * amp_guess, x0_guess + 1, 0.175, y0_guess + 1, 0.175, offset_guess * 1.2]

            # Perform fit
            res = least_squares(residuals, p0, args=(x_flat, y_flat, z_flat), bounds=(lb, ub))
            popt = res.x
            amp_fit, x0_fit, sigx_fit, y0_fit, sigy_fit, offset_fit = popt
            brightness_fit = 2 * np.pi * amp_fit * sigx_fit * sigy_fit / pix_size_um**2
            brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * offset_fit

            if brightness_fit > 1e9 or brightness_fit < 50:
                print(f"Excluded peak for brightness {brightness_fit:.2e}")
                continue
            if sigx_fit > sig_threshold or sigy_fit > sig_threshold:
                print(f"Excluded peak for size {sigx_fit:.2f} um x {sigy_fit:.2f} um")
                continue

            # Note: coordinates are already RELATIVE to cropped image
            results.append({
                'x_pix': center_x_refined,
                'y_pix': center_y_refined,
                'x_um': x0_fit,
                'y_um': y0_fit,
                'amp_fit': amp_fit,
                'sigx_fit': sigx_fit,
                'sigy_fit': sigy_fit,
                'brightness_fit': brightness_fit,
                'brightness_integrated': brightness_integrated
            })

        except RuntimeError:
            continue

    df = pd.DataFrame(results)
    return df, image_data_cps
    
def gaussian(x, amp, mu, sigma):
  return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_brightness(image_data_cps, df, show_fits = True, plot_brightness_histogram = False, normalization = False, pix_size_um = 0.1):

    fig_width, fig_height = 5, 5
    
    scale = fig_width / 5  

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if normalization:
        normalization = LogNorm()
    else:
        normalization = None
    im = ax.imshow(image_data_cps + 1, cmap='magma', norm=normalization, origin='lower') #LogNorm()
    ax.tick_params(axis='both',length=0, labelleft=False, labelright=False, labeltop=False, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 10*scale)
    cbar.set_label('pps', fontsize=10*scale)  

    if show_fits:
        for _, row in df.iterrows():
            x_px = row['x_pix']
            y_px = row['y_pix']
            brightness_kpps = row['brightness_fit'] / 1000
            radius_px = 2 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um
    
            circle = Circle((x_px, y_px), radius_px, color='white', fill=False, linewidth=1*scale, alpha=0.7)
            ax.add_patch(circle)
    
            ax.text(x_px + 7.5, y_px + 7.5, f"{brightness_kpps:.1f} kpps",
                    color='white', fontsize=7*scale, ha='center', va='center')

    #ax.set_xlabel('x (px)', fontsize = 10*scale)
    #ax.set_ylabel('y (px)', fontsize = 10*scale)
    plt.tight_layout()
    HWT_aesthetic()
    return fig


def plot_histogram(df, save_as_svg=False, min_val=None, max_val=None):
    """Plots the brightness histogram with a gaussian fit."""
    fig_width, fig_height = 4, 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scale = fig_width / 5

    brightness_vals = df['brightness_fit'].values

    # Apply min/max filtering if specified
    if min_val is not None and max_val is not None:
        brightness_vals = brightness_vals[(brightness_vals >= min_val) & (brightness_vals <= max_val)]

    # Ensure the filtered data is valid
    if len(brightness_vals) == 0:
        st.warning("No data in selected brightness range.")
        return fig

    # Use the min/max values to define histogram bin edges
    bins = np.linspace(min_val, max_val, 20)

    counts, edges, _ = ax.hist(brightness_vals, bins=bins, color='#88CCEE', edgecolor='#88CCEE')
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    p0 = [np.max(counts), np.mean(brightness_vals), np.std(brightness_vals)]
    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        mu, sigma = popt[1], popt[2]
        x_fit = np.linspace(edges[0], edges[-1], 500)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='black', linewidth=0.75, linestyle='--', label=f"μ = {mu:.0f} ± {sigma:.0f} pps")
        ax.legend(fontsize=10 * scale)
    except RuntimeError:
        st.warning("Gaussian fit failed.")

    ax.set_xlabel("Brightness (pps)", fontsize=10 * scale)
    ax.set_ylabel("Count", fontsize=10 * scale)
    ax.tick_params(axis='both', labelsize=10 * scale, width=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    HWT_aesthetic()
    plt.tight_layout()
    return fig





def sort_UCNP_dye_sifs(directory, signal_ucnp=976, signal_dye=638):
    files = [f for f in os.listdir(directory) if f.endswith('.sif')]
    ucnp_files = []
    dye_files = []

    for f in files:
        full_path = os.path.join(directory, f)
        has_ucnp = str(signal_ucnp) in f
        has_dye = str(signal_dye) in f

        if has_ucnp and not has_dye:
            ucnp_files.append(full_path)
        elif has_dye and not has_ucnp:
            dye_files.append(full_path)
        elif has_ucnp and has_dye:
            print(f"Warning: file contains both excitation numbers → {f}")
        else:
            print(f"Warning: file contains neither excitation number → {f}")

    return ucnp_files, dye_files


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def match_ucnp_dye_files(ucnps, dyes):
    # 1. deterministic sorting
    ucnps_sorted = sorted(ucnps, key=natural_sort_key)
    dyes_sorted = sorted(dyes, key=natural_sort_key)

    # 2. build dye lookup
    dye_map = {}
    for f in dyes_sorted:
        m = re.search(r'(\d+)\.sif$', f)
        if m:
            dye_map[int(m.group(1))] = f

    pairs = []
    warnings = []
    matched_dyes = set()

    # 3. for each UCNP, try forward then backward, skipping used dyes
    for ucnp_file in ucnps_sorted:
        m = re.search(r'(\d+)\.sif$', ucnp_file)
        if not m:
            warnings.append(f"Could not parse UCNP index from {ucnp_file}")
            continue
        uidx = int(m.group(1))

        # candidate indices
        cand = []
        if (uidx + 1) in dye_map and (uidx + 1) not in matched_dyes:
            cand.append(uidx + 1)
        if (uidx - 1) in dye_map and (uidx - 1) not in matched_dyes:
            cand.append(uidx - 1)

        if not cand:
            # no unmatched forward/backward
            expected = []
            if (uidx+1) in dye_map: expected.append(str(uidx+1))
            if (uidx-1) in dye_map: expected.append(str(uidx-1))
            if expected:
                warnings.append(
                    f"Both candidate dyes {', '.join(expected)} for UCNP {os.path.basename(ucnp_file)} "
                    "are already matched to other UCNPs."
                )
            else:
                warnings.append(
                    f"No Dye file “{uidx+1}.sif or {uidx-1}.sif” found for UCNP {os.path.basename(ucnp_file)}"
                )
        else:
            # pick forward first if present, else backward
            chosen_idx = cand[0]
            pairs.append((ucnp_file, dye_map[chosen_idx]))
            matched_dyes.add(chosen_idx)

    # 4. report any warnings
    for w in warnings:
        print("Warning:", w)

    return pairs

    
def coloc_subplots(ucnps, dyes, df_dict, colocalization_radius=2, show_fits=True, save_SVG = False):
    n_pairs = min(len(ucnps), len(dyes))
    #per_file_matched = []
    all_matched_pairs = []


    required_cols = ['x_pix', 'y_pix', 'sigx_fit', 'sigy_fit', 'brightness_fit']
    if len(ucnps) != len(dyes):
      print(f"Warning: UCNP files ({len(ucnps)}) and Dye files ({len(dyes)}) have different lengths.")

    if len(ucnps) != len(dyes):
        print(f"Warning: UCNP files ({len(ucnps)}) and Dye files ({len(dyes)}) have different lengths.")


    ucnp_indices = [int(re.search(r'(\d+)\.sif$', f).group(1)) for f in ucnps]
    dye_indices = [int(re.search(r'(\d+)\.sif$', f).group(1)) for f in dyes]



    pairs = match_ucnp_dye_files(ucnps, dyes)

    # Flatten pairs into two separate lists for the rest of the function
    ucnps = [pair[0] for pair in pairs]
    dyes = [pair[1] for pair in pairs]
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 6.5 * n_pairs))

    if n_pairs == 1:
        axes = np.array([axes])

    for i in range(n_pairs):
        per_file_matched = []
        ucnp_file = ucnps[i]
        dye_file = dyes[i]
        ucnp_file_name = os.path.basename(ucnp_file)
        dye_file_name = os.path.basename(dye_file)

        # Extract file IDs (numbers before .sif)
        ucnp_file_id = re.search(r'(\d+)\.sif$', ucnp_file_name)
        dye_file_id = re.search(r'(\d+)\.sif$', dye_file_name)

        print(f"Pair {i+1}:")
        print(f"  UCNP file: {ucnp_file_name}")
        print(f"  Dye file: {dye_file_name}")

        matched_ucnp = set()
        matched_dye = set()

        if ucnp_file_id and dye_file_id:
          ucnp_file_id = ucnp_file_id.group(1)
          dye_file_id = dye_file_id.group(1)

        ucnp_df, ucnp_img = df_dict[ucnp_file]
        dye_df, dye_img = df_dict[dye_file]

        # Determine if each DataFrame has the required columns
        ucnp_has_fit = all(col in ucnp_df.columns for col in required_cols)
        dye_has_fit = all(col in dye_df.columns for col in required_cols)

        # Initialize colocalization arrays
        colocalized_ucnp = np.zeros(len(ucnp_df), dtype=bool) if ucnp_has_fit else None
        colocalized_dye = np.zeros(len(dye_df), dtype=bool) if dye_has_fit else None

        # Perform colocalization only if both DataFrames have the required columns and show_fits is True
        if show_fits and ucnp_has_fit and dye_has_fit:
            for idx_ucnp, row_ucnp in ucnp_df.iterrows():
                x_ucnp, y_ucnp = row_ucnp['x_pix'], row_ucnp['y_pix']

                # compute distances to all *unmatched* dyes
                mask = ~dye_df.index.isin(matched_dye)
                dx = dye_df.loc[mask, 'x_pix'] - x_ucnp
                dy = dye_df.loc[mask, 'y_pix'] - y_ucnp
                distances = np.hypot(dx, dy)

                if distances.min() <= colocalization_radius:
                    best_dye_idx = distances.idxmin()
                    matched_ucnp.add(idx_ucnp)
                    matched_dye.add(best_dye_idx)

                    row_dye = dye_df.loc[best_dye_idx]

                    colocalized_ucnp[idx_ucnp] = True
                    closest_idx = distances.idxmin()
                    row_dye = dye_df.loc[closest_idx]
                    idx_dye = dye_df.index.get_loc(closest_idx)
                    colocalized_dye[idx_dye] = True


                    #update df with relevant fields
                    per_file_matched.append({
                        'ucnp_file': ucnp_file_id,
                        'dye_file': dye_file_id,
                        'ucnp_x': x_ucnp,
                        'ucnp_y': y_ucnp,
                        'ucnp_brightness': row_ucnp['brightness_fit'],
                        'dye_x': row_dye['x_pix'],
                        'dye_y': row_dye['y_pix'],
                        'dye_brightness': row_dye['brightness_fit'],
                        'distance': distances[closest_idx],
                        'x_um_ucnp': row_ucnp['x_um'],
                        'y_um_ucnp': row_ucnp['y_um'],
                        'sigx_ucnp': row_ucnp['sigx_fit'],
                        'sigy_ucnp': row_ucnp['sigy_fit'],
                        'x_um_dye': row_dye['x_um'],
                        'y_um_dye': row_dye['y_um'],
                        'sigx_dye': row_dye['sigx_fit'],
                        'sigy_dye': row_dye['sigy_fit']
                    })

            percent_ucnp_coloc = 100 * np.sum(colocalized_ucnp) / len(ucnp_df) if len(ucnp_df) > 0 else 0
            percent_dye_coloc = 100 * np.sum(colocalized_dye) / len(dye_df) if len(dye_df) > 0 else 0
        else:
            percent_ucnp_coloc = percent_dye_coloc = None
            matched_pairs = []

        all_matched_pairs.extend(per_file_matched)



        matched_pairs_df = pd.DataFrame(per_file_matched)


        # --- Plot UCNP ---
        ax_ucnp = axes[i, 0]
        im_ucnp = ax_ucnp.imshow(ucnp_img + 1, cmap='magma', origin='lower', norm=LogNorm())
        plt.colorbar(im_ucnp, ax=ax_ucnp, label='pps', fraction=0.046, pad=0.04)

        ucnp_basename = os.path.basename(ucnp_file)
        match = re.search(r'(\d+)\.sif$', ucnp_basename)
        file_number = match.group(1) if match else '?'
        wrapped_ucnp_basename = "\n".join(textwrap.wrap(ucnp_basename, width=35))


        if show_fits and ucnp_has_fit:
            for is_coloc, (_, row) in zip(colocalized_ucnp, ucnp_df.iterrows()):
                color = 'lime' if is_coloc else 'white'
                radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / 0.1
                circle = Circle((row['x_pix'], row['y_pix']), radius_px, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax_ucnp.add_patch(circle)
                ax_ucnp.text(row['x_pix'] + 7.5, row['y_pix'] + 7.5,
                            f"{row['brightness_fit']/1000:.1f} kpps",
                            color=color, fontsize=8, ha='center', va='center')

            if percent_ucnp_coloc is not None:
                ax_ucnp.set_title(f"UCNP sif# {file_number}\n{wrapped_ucnp_basename}\n{percent_ucnp_coloc:.1f}% colocalized")
            else:
                ax_ucnp.set_title(f"UCNP sif# {file_number}\n{wrapped_ucnp_basename}\n[No colocalization data]")
        else:
            ax_ucnp.set_title(f"UCNP\n{wrapped_ucnp_basename}")

        ax_ucnp.set_xlabel('X (px)')
        ax_ucnp.set_ylabel('Y (px)')

        # --- Plot Dye ---
        ax_dye = axes[i, 1]
        im_dye = ax_dye.imshow(dye_img + 1, cmap='magma', origin='lower', norm=LogNorm())
        plt.colorbar(im_dye, ax=ax_dye, label='pps', fraction=0.046, pad=0.04)

        dye_basename = os.path.basename(dye_file)
        wrapped_dye_basename = "\n".join(textwrap.wrap(dye_basename, width=35))


        if show_fits and dye_has_fit:
            for is_coloc, (_, row) in zip(colocalized_dye, dye_df.iterrows()):
                color = 'lime' if is_coloc else 'white'
                radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um
                circle = Circle((row['x_pix'], row['y_pix']), radius_px, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax_dye.add_patch(circle)
                ax_dye.text(row['x_pix'] + 7.5, row['y_pix'] + 7.5,
                            f"{row['brightness_fit']/1000:.1f} kpps",
                            color=color, fontsize=8, ha='center', va='center')

            match = re.search(r'(\d+)\.sif$', wrapped_dye_basename)
            file_number = match.group(1) if match else '?'
            ax_dye.set_title(f"Dye sif# {file_number}\n{wrapped_dye_basename}\n{percent_dye_coloc:.1f}% colocalized")
        else:
            ax_dye.set_title(f"Dye\n{wrapped_dye_basename}")

        ax_dye.set_xlabel('X (px)')
        ax_dye.set_ylabel('Y (px)')


    compiled_matched_df = pd.DataFrame(all_matched_pairs)
    plt.tight_layout()

    if save_SVG:
        expt_name = os.path.basename(file_path.rstrip('/'))
        expt_name = re.sub(r'[^\w\-_.]', '_', expt_name)
        todaydate = date.today().strftime("%Y%m%d")
        filename = f"{expt_name}_{todaydate}.svg"

        plt.savefig(filename)
        print(f"Saved in Colab VM as: {filename}")
        files.download(filename)  # This will prompt a browser download
    plt.show()
    return compiled_matched_df

def extract_subregion(image, x0, y0, radius_pix):
    x_start = int(max(x0 - radius_pix, 0))
    x_end = int(min(x0 + radius_pix + 1, image.shape[1]))
    y_start = int(max(y0 - radius_pix, 0))
    y_end = int(min(y0 + radius_pix + 1, image.shape[0]))
    return image[y_start:y_end, x_start:x_end], x_start, y_start

def gaussian2d(xy, amp, x0, sigma_x, y0, sigma_y, offset):
    x, y = xy
    return (amp * np.exp(-((x - x0)**2)/(2*sigma_x**2)) *
                 np.exp(-((y - y0)**2)/(2*sigma_y**2)) + offset).ravel()



