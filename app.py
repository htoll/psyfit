import streamlit as st
from utils import integrate_sif, plot_brightness, sort_UCNP_dye_sifs, natural_sort_key, match_ucnp_dye_files, coloc_subplots, extract_subregion, gaussian2d, HWT_aesthetic, plot_histogram
import sif_parser

from skimage.feature import peak_local_max
from skimage.feature import blob_log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from datetime import date
import os
import io

# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

import streamlit as st
import os
st.set_page_config(layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
tool = st.sidebar.radio("Select a tool:", [
    "Analyze single SIF",
    "Analyze Colocalization Set",
    "Batch Convert SIFs",
    "Visualize Data"
])
col1, col2 = st.columns([1, 2])

# Tool: Analyze single SIF
if tool == "Analyze single SIF":
    analyze_single_sif.run()

# Tool: Analyze Colocalization Set
elif tool == "Colocalization Set":
    st.header("Colocalization Set")
    st.info("This feature is under construction — implement logic here.")

# Tool: Batch Convert SIFs
elif tool == "Batch Convert SIFs":
    st.header("Batch Convert SIFs")
    st.info("This feature is under construction — implement logic here.")

# Tool: Visualize Data
elif tool == "Visualize Data":
    st.header("Visualize Data")
    st.info("This feature is under construction — implement logic here.")




