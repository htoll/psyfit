# -*- coding: utf-8 -*-
"""conc_monomer_code_HWT_250506.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R5TRAtMPwPY9Th96OAaZ65rBjZXcDDtB
"""

#imports
try:
    import sif_parser
except ImportError:
    !pip install sif_parser
    import sif_parser
import numpy as np
from datetime import datetime, timedelta
import os
from os import listdir, mkdir
import re

from google.colab import drive
import sif_parser

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets

from skimage.feature import peak_local_max
from skimage.feature import blob_log



import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import scipy

#mount files from drive
drive.mount('/content/drive/', force_remount=True)


#Past files:
#HWT05_055A: /content/drive/Shareddrives/PengLab_Data_2025/Microscopy/HWT/202505/20250501/HWT05_055A_Tm02Yb98@DNA_1xPBS_976nm_700mA_638nm_80mA

#global variables:
pix_size_um = 0.1
sig_threshold = 0.3 #threshold to remove psfs greater than this sigma

# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

"""# Functions"""

def integrate_sif(sif, threshold=1, region='all', signal='UCNP'):
    if not sif.endswith('.sif'):
        return pd.DataFrame(), None
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

        # Prepare grid
        y_indices, x_indices = np.indices(sub_img_fine.shape)
        x_coords = (x_indices + x0_idx_fine) * pix_size_um
        y_coords = (y_indices + y0_idx_fine) * pix_size_um

        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        z_flat = sub_img_fine.ravel()

        # Initial guess
        amp_guess = np.max(sub_img_fine)
        offset_guess = np.min(sub_img_fine)
        x0_guess = center_x_refined * pix_size_um
        y0_guess = center_y_refined * pix_size_um
        sigma_guess = 0.15
        p0 = [amp_guess, x0_guess, sigma_guess, y0_guess, sigma_guess, offset_guess]

        # Fit
        try:
            popt, _ = curve_fit(gaussian2d, (x_flat, y_flat), z_flat, p0=p0)
            amp_fit, x0_fit, sigx_fit, y0_fit, sigy_fit, offset_fit = popt
            brightness_fit = 2 * np.pi * amp_fit * sigx_fit * sigy_fit / pix_size_um**2
            brightness_integrated = np.sum(sub_img_fine) - sub_img_fine.size * offset_fit


            if sigx_fit <= 0 or sigy_fit <= 0:
              continue
            if brightness_fit > 1e9:
                #print(f"Excluded peak for brightness {brightness_fit:.2e}")
                continue
            if sigx_fit > sig_threshold or sigy_fit > sig_threshold:
                #print(f"Excluded peak for size {sigx_fit:.2f} um x {sigy_fit:.2f} um")
                continue
            if brightness_fit < 0: #lower bound
                #print(f"Excluded peak for brightness {brightness_fit:.2e}")
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

def plot_brightness(image_data_cps, df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image_data_cps + 1, cmap='magma', norm=LogNorm(), origin='lower')
    plt.colorbar(im, ax=ax, label='pps', fraction=0.046, pad=0.04)

    for _, row in df.iterrows():
        x_px = row['x_pix']
        y_px = row['y_pix']
        brightness_kpps = row['brightness_fit'] / 1000
        radius_px = 2 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um

        circle = Circle((x_px, y_px), radius_px, color='white', fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)

        ax.text(x_px + 7.5, y_px + 7.5, f"{brightness_kpps:.1f} kpps",
                color='white', fontsize=10, ha='center', va='center')

    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    if ax is None:
        plt.show()

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

def conc_calculator(dilution, particles_per_fov, well_vol, well_diam = 3):
  '''
  Estimates the concentration of a solution based on the average number of particles per field of view
  Inputs:
    dilution: difference in concentration between stock and measured sample. Ex: 1 to 10 dilution is 1/10
    particles_per_fov: avg number of fit particles across frames
    well_vol: volume added to well in uL. Assumes all particles adhere to slide uniformly
    well_diam: diameter of well in microns

  Returns estimated concentration of the stock solution in moles/liter

  '''
  area_fov = np.pi * 12.8**2 #um^2
  area_well = np.pi * (well_diam * 1000 /2)**2 #um^2

  ratio_well_to_fov = area_well / area_fov
  particles_well = ratio_well_to_fov * particles_per_fov #num particles in the entire well
  conc_well = (particles_well / (well_vol / 1e6)) / scipy.constants.N_A #conc in M of well
  concentration_stock = conc_well / dilution

  return concentration_stock

def convert_prefixM(concentration):
    prefix = ''
    divisor = 1
    if concentration >= 1:
        prefix = 'M'
    elif concentration >= 10**-3:
        prefix = 'mM'
        divisor = 10**-3
    elif concentration >= 10**-6:
        prefix = 'uM'
        divisor = 10**-6
    elif concentration >= 10**-9:
        prefix = 'nM'
        divisor = 10**-9
    elif concentration >= 10**-12:
        prefix = 'pM'
        divisor = 10**-12
    elif concentration >= 10**-15:
        prefix = 'fM'
        divisor = 10**-15
    else:
        prefix = 'aM'
        divisor = 10**-18
    return [concentration / divisor, prefix]

test_conc = conc_calculator(dilution = 1/(1*100), particles_per_fov = 500, well_vol = 5, well_diam = 3)

print(convert_prefixM(test_conc))

"""# Implementation

"""

#file_path = r'/content/drive/Shared drives/PengLab_Data_2025/Microscopy/HWT/202506/20250609/HWT05_097_1to10k_Er01Yb05N3_976nm_1000mA'
file_path = r'/content/drive/Shared drives/PengLab_Data_2025/Microscopy/HWT/202507/20250731/Tm06Yb94_PMAO_KM_1to1k_976nm1500mA'

print(os.listdir(file_path))

file = os.path.join(file_path, 'Tm06Yb94_PMAO_KM_1to100k_976nm1500mA_1.sif')
region = '1'
threshold = 0.05
signal = 'UCNP'
# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

df, image_data = integrate_sif(file, threshold = threshold, region = region, signal = signal)
plot_brightness(image_data, df)

#calculate # nps per PSF


compiled_df = pd.DataFrame()
for sif in os.listdir(file_path): #extract sifs
  #print(sif)
  df, image_data_cps = integrate_sif(os.path.join(file_path, sif), threshold = threshold, region = region, signal = signal)
  compiled_df = pd.concat([compiled_df, df])

num_psf = len(compiled_df)
num_frames = len(os.listdir(file_path))
nps_per_fov =  num_psf / num_frames

conc = conc_calculator(dilution = 1/(1*100), particles_per_fov = nps_per_fov, well_vol = 5, well_diam = 3)
print(f'Calculated NP per fov: {nps_per_fov}')

converted_conc = convert_prefixM(conc)
print(f'Calculated concentration: {converted_conc[0]:.2f} {converted_conc[1]}')

#plot all brightnesses as histogram and fit gmm

#compiled_df.head()
brightnesses = compiled_df['brightness_fit']

percentile_lower = np.percentile(brightnesses, 5)
percentile_upper = np.percentile(brightnesses, 95)
cropped_brightnesses = brightnesses[(brightnesses >= percentile_lower) & (brightnesses <= percentile_upper)]
plt.hist(cropped_brightnesses, bins=100, alpha=0.5, color='red')

plt.xlim(percentile_lower, percentile_upper)
plt.xlabel('Brightness (cps)')
plt.ylabel('Counts')
plt.show()
print(len(brightnesses))

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- data ---
brightnesses = compiled_df['brightness_fit'].values

# Optionally crop percentiles
percentile_lower = np.percentile(brightnesses, 5)
percentile_upper = np.percentile(brightnesses, 95)
cropped_brightnesses = brightnesses[(brightnesses >= percentile_lower) & (brightnesses <= percentile_upper)]

# --- reshape for sklearn ---
data = cropped_brightnesses.reshape(-1, 1)

# --- fit GMM ---
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(data)

mu = gmm.means_.flatten()
sigma = np.sqrt(gmm.covariances_.flatten())
A = gmm.weights_

print("Fitted means:", mu)
print("Fitted std devs:", sigma)
print("Fitted amplitudes:", A)

# --- plot ---
x = np.linspace(min(cropped_brightnesses), max(cropped_brightnesses), 500)
bin_counts, bin_edges = np.histogram(cropped_brightnesses, bins=50)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]
scale_factor = len(cropped_brightnesses) * bin_width

plt.bar(bin_centers, bin_counts, width=bin_width, alpha=0.5)

# plot total PDF
pdf_total = np.zeros_like(x)
for k in range(3):
    pdf_k = A[k] * norm.pdf(x, loc=mu[k], scale=sigma[k])
    plt.plot(x, pdf_k * scale_factor, linestyle='--', label=f'Component {k+1}')
    pdf_total += pdf_k

plt.plot(x, pdf_total * scale_factor, color='red', lw=2, label='Total fit')

plt.xlabel('Brightness (cps)')
plt.ylabel('Counts')
plt.legend()
plt.show()

#percent of psf's in each gaussian:
# --- assign each point to most likely component ---
labels = gmm.predict(data)  # shape (N,)

# --- count occurrences ---
unique, counts = np.unique(labels, return_counts=True)

# --- convert counts to percentages ---
percentages = counts / len(data) * 100

# --- bar plot ---
import matplotlib.pyplot as plt

plt.bar([f"Component {i+1}" for i in unique], percentages)
plt.ylabel('Percentage of data points (%)')
plt.show()

for i, p in zip(unique, percentages):
    print(f"Component {i+1}: {p:.1f}%")

def plot_quantized_histogram(brightnesses, mu, color, label=None, xlim=(0.5, 14.5), ylim=(0, 100)):
    """
    Plots a histogram of quantized nanoparticle counts based on brightness and GMM means.

    Parameters:
    - brightnesses: np.ndarray of raw brightness values
    - mu: np.ndarray of GMM means (flattened)
    - color: color for the histogram bars (hex or named string)
    - label: legend label
    - xlim: tuple for x-axis limits
    - ylim: tuple for y-axis limits
    """

    # --- Identify single-particle brightness ---
    single_index = np.argmin(mu)
    unit_brightness = mu[single_index]

    # --- Estimate quantized NP counts ---
    estimated_counts = brightnesses / unit_brightness
    estimated_counts_rounded = np.round(estimated_counts)
    estimated_counts_rounded = estimated_counts_rounded[estimated_counts_rounded != 0] #exclude 0's


    # --- Crop to 1st–99th percentile ---
    lower_bound = np.percentile(estimated_counts_rounded, 1)
    upper_bound = np.percentile(estimated_counts_rounded, 99)
    in_range = (estimated_counts_rounded >= lower_bound) & (estimated_counts_rounded <= upper_bound)
    valid_counts = estimated_counts_rounded[in_range]

    # --- Define bins and calculate percentages ---
    bins = np.arange(np.floor(lower_bound), np.ceil(upper_bound) + 1) - 0.5
    counts, edges = np.histogram(valid_counts, bins=bins)
    percentages = 100 * counts / len(valid_counts)

    # --- Plot ---
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    plt.bar(bin_centers, percentages, width=1.0, color=color, edgecolor='black', alpha=0.5, label=label)


# === Example Usage ===

plt.figure(figsize=(8, 5))

# First dataset
plot_quantized_histogram(brightnesses=brightnesses, mu=mu, color='#fa8775', label=f'HWT05_097, {convert_prefixM(conc)[0]:.2f} {convert_prefixM(conc)[1]}')

# Add second dataset
#plot_quantized_histogram(brightnesses=brightnesses2, mu=mu2, color='#0000ff', label='250g, 1 hr')

# Labels and aesthetics
plt.xlabel('Estimated # of Nanoparticles per PSF')
plt.ylabel('% of PSFs')
plt.title('Quantized Nanoparticle Count per PSF')
plt.xticks(range(1, 15))
plt.xlim([-0.5, 14.5])
plt.legend()
plt.tight_layout()
plt.show()

datasets = [
    {
        "path": "/content/drive/Shared drives/PengLab_Data_2025/Microscopy/HWT/202506/20250609/HWT05_099A_1xPBS1mMNaFovernight_1to1k_976nm_1000mA",
        "label": "1x PBS, 1mM NaF (05-099A)",
        "color": "#fa8775",
        "dilution": 1 / (1*10**3)
    },
    {
        "path": "/content/drive/Shared drives/PengLab_Data_2025/Microscopy/HWT/202506/20250609/HWT05_099B_1xTHPTABufferovernight_1to1k_976nm_1000mA",
        "label": "1x THPTA (05_099B)",
        "color": "#377eb8",
        "dilution": 1 / (1*10**3)
    },
    # {
    #     "path": "/content/drive/Shareddrives/PengLab_Data_2025/Microscopy/HWT/202506/20250606/HWT05_095A_Er01Yb05_1p5Scale_250g1hr_1to20k_3uL_3mmwell_976nm_500mA",
    #     "label": "05_091A, 250g for 1 hr",
    #     "color": "#4daf4a",
    #     "dilution": 1 / (2*10**4)
    # }
]


# Constants that are shared
region = '1'
signal = 'UCNP'

# === Plot all datasets ===
plt.figure(figsize=(8, 5))

for dataset in datasets:
    file_path = dataset['path']
    label = dataset['label']
    color = dataset['color']
    dilution = dataset['dilution']

    compiled_df = pd.DataFrame()

    sifs = [f for f in os.listdir(file_path) if f.endswith('.sif')]

    for sif in sifs: #extract sifs
      #print(sif)
      df, image_data_cps = integrate_sif(os.path.join(file_path, sif), threshold = threshold, region = region, signal = signal)
      compiled_df = pd.concat([compiled_df, df])

    num_psf = len(compiled_df)
    num_frames = len(os.listdir(file_path))
    nps_per_fov =  num_psf / num_frames

    if compiled_df.empty:
        print(f"Warning: No valid data found in {file_path}")
        continue

    conc = conc_calculator(dilution = dilution, particles_per_fov = nps_per_fov, well_vol = 3, well_diam = 3)
    molar_conc = convert_prefixM(conc)
    brightnesses = compiled_df['brightness_fit'].values.reshape(-1, 1)
    print(f"Num psfs: {len(brightnesses)}")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
    gmm.fit(brightnesses)
    mu = np.sort(gmm.means_.flatten())

    full_label = f"{label} ({molar_conc[0]:.2f} {molar_conc[1]})"
    plot_quantized_histogram(brightnesses.flatten(), mu, color, full_label)

# Final plot styling
plt.xlabel('Estimated # of Nanoparticles per PSF')
plt.ylabel('% of PSFs')
plt.title('Quantized Nanoparticle Count per PSF')
#plt.xticks(range(1, 15))
plt.xlim([0.5, 50])
plt.legend()
plt.tight_layout()
plt.show()

#plot surface area as function of shell thickness

import matplotlib.pyplot as plt

# Base dimensions
base_edge = 34.5 / 2
height = 47.71

def hex_prism_sa(base_edge, height):
    sa = (6 * base_edge * height) + (3 * (3 ** 0.5) * base_edge ** 2)
    return sa

shell_thicknesses = range(0, 30)  # nm

new_surface_area = []
for thickness in shell_thicknesses:
    new_surface_area.append(hex_prism_sa(base_edge + thickness, height))

# Compute % increase in surface area
percent_increase = [(sa - new_surface_area[0]) / new_surface_area[0] * 100 for sa in new_surface_area]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Original surface area plot
ax[0].plot(shell_thicknesses, new_surface_area, label="Surface Area", color="blue")
ax[0].set_xlabel("Shell Thickness (nm)")
ax[0].set_ylabel("Surface Area (nm²)")
ax[0].set_title("Surface Area vs. Shell Thickness")
ax[0].grid(True)

# % Increase plot
ax[1].plot(shell_thicknesses, percent_increase, label="% Increase", color="green")
ax[1].set_xlabel("Shell Thickness (nm)")
ax[1].set_ylabel("Percentage Increase (%)")
ax[1].set_title("Percentage Increase in Surface Area")
ax[1].grid(True)

plt.tight_layout()
plt.show()