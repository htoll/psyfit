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
from matplotlib.colors import Normalize

import seaborn as sns

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from datetime import date

import streamlit as st
import io
import re
import os
import textwrap

from utils import HWT_aesthetic

def gaussian(x, amp, mu, sigma):
  return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
def plot_histogram(df, min_val=None, max_val=None, num_bins=20, thresholds=None):
    """
    Plots the brightness histogram with a Gaussian fit and optional vertical thresholds.
    
    Args:
        df (pd.DataFrame): DataFrame containing brightness data.
        min_val (float, optional): Minimum brightness value for the histogram.
        max_val (float, optional): Maximum brightness value for the histogram.
        num_bins (int, optional): Number of bins for the histogram.
        thresholds (list, optional): A list of numerical values to plot as vertical lines.
    """
    fig_width, fig_height = 4, 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scale = fig_width / 5

    brightness_vals = df['brightness_fit'].values

    # Apply min/max filtering if specified
    if min_val is not None and max_val is not None:
        brightness_vals = brightness_vals[(brightness_vals >= min_val) & (brightness_vals <= max_val)]

    # If the filtered data is empty, return an empty figure
    if len(brightness_vals) == 0:
        return fig

    # Use the min/max values to define histogram bin edges
    bins = np.linspace(min_val, max_val, num_bins)

    counts, edges, _ = ax.hist(brightness_vals, bins=bins, color='#88CCEE', edgecolor='#88CCEE', alpha=0.7)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Gaussian fit
    mu, sigma = None, None
    p0 = [np.max(counts), np.mean(brightness_vals), np.std(brightness_vals)]
    try:
        popt = p0
        mu, sigma = popt[1], popt[2]
        x_fit = np.linspace(edges[0], edges[-1], 500)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='black', linewidth=0.75, linestyle='--', label=r"μ = {mu:.0f} ± {sigma:.0f} pps".format(mu=mu, sigma=sigma))
        ax.legend(fontsize=10 * scale)
    except RuntimeError:
        pass  # Fail gracefully if fit doesn't converge

    palette = HWT_aesthetic()
    region_colors = palette[:4]

    # Draw shaded background regions first
    if thresholds:
        all_bounds = [min_val] + sorted(thresholds) + [max_val]
        for i in range(len(all_bounds) - 1):
            ax.axvspan(
                all_bounds[i],
                all_bounds[i + 1],
                color=region_colors[i % len(region_colors)],
                alpha=0.2,
                zorder=0  # optional: send even further back
            )

    # Now draw histogram bars on top
    counts, edges, _ = ax.hist(
        brightness_vals,
        bins=np.linspace(min_val, max_val, num_bins),
        color='#88CCEE',
        edgecolor='#88CCEE',
        alpha=0.7,
        zorder=1
    )

    ax.set_xlabel("Brightness (pps)", fontsize=10 * scale)
    ax.set_ylabel("Count", fontsize=10 * scale)
    ax.tick_params(axis='both', labelsize=10 * scale, width=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    HWT_aesthetic()
    plt.tight_layout()
    return fig, mu, sigma
