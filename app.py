import streamlit as st
import sys
import os
import io


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tools import analyze_single_sif, batch_convert, colocalization, monomers, delaunayJFS, plot_csv
import tools.spherical_tem as spherical_tem

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



# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

import streamlit as st
import os
st.set_page_config(layout="wide")

# Sidebar navigation
st.sidebar.title("Tools")
tool = st.sidebar.radio("Analyze:", [
    "Batch Convert",
    "Brightness",
    "Delaunay Colocalization",
    "Dye Colocalization",
    "Monomer + Conc Estimation",
    "Plot CSVs",
    "Spherical NP TEM"
    
])
col1, col2 = st.columns([1, 2])

if tool == "Brightness":
    analyze_single_sif.run()


# Tool: Analyze Colocalization Set
elif tool == "Dye Colocalization":
    colocalization.run()

# Tool: Batch Convert SIFs
elif tool == "Batch Convert":
    batch_convert.run()

elif tool == 'Monomer + Conc Estimation':
    monomers.run()

elif tool == 'Delaunay Colocalization':
    delaunayJFS.run()

elif tool == "Plot CSVs":
    plot_csv.run()

elif tool =="Spherical NP TEM":
    spherical_tem.run()


