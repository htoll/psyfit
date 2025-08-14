import streamlit as st
import sys
import os
import io


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tools import analyze_single_sif, batch_convert, colocalization, monomers, delaunayJFS
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
    "Monomer + Conc Estimation",
    "UNDER CONSTRUCTION Colocalization Set",
    "Delaunay Colocalization"
    
])
col1, col2 = st.columns([1, 2])

if tool == "Brightness":
    analyze_single_sif.run()


# Tool: Analyze Colocalization Set
elif tool == "UNDER CONSTRUCTION Colocalization Set":
    colocalization.run()

# Tool: Batch Convert SIFs
elif tool == "Batch Convert":
    batch_convert.run()

elif tool == 'Monomer + Conc Estimation':
    monomers.run()
    
elif tool == 'Delaunay Colocalization':
    #delaunayJFS.run() #HWT250814, was givign AxiosError: timeout exceeded on other tools upon file upload, terminal output:
        #  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  
        
        #   nner/exec_code.py:128 in exec_func_with_error_handling                        
        
                                                                                        
        
        #   /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  
        
        #   nner/script_runner.py:669 in code_to_exec                                     
        
                                                                                        
        
        #   /mount/src/psyfit/app.py:69 in <module>                                       
        
                                                                                        
        
        #     66 │   monomers.run()                                                       
        
        #     67                                                                          
        
        #     68 elif tool == 'Delaunay Colocalization':                                  
        
        #   ❱ 69 │   delaunayJFS.run()                                                    
        
        #     70                                                                          
        
        #     71                                                                          
        
        #     72                                                                          
        
                                                                                        
        
        #   /mount/src/psyfit/tools/delaunayJFS.py:326 in run                             
        
                                                                                        
        
        #     323 │   run()                                                               
        
        #     324                                                                         
        
                                                                                        
        
        #   /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/state/se  
        
        #   ssion_state_proxy.py:132 in __getattr__                                       
        
        # ────────────────────────────────────────────────────────────────────────────────
        
        # AttributeError: st.session_state has no attribute "active_quads_processed". Did 
        
        # you forget to initialize it? More info: 
        
        # https://docs.streamlit.io/develop/concepts/architecture/session-state#initializa
        
        # tion
        
        # [17:39:08] 🔄 Updated app!
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
        
        # <tifffile.TiffTag 5033 @530884> coercing invalid ASCII to bytes, due to UnicodeDecodeError('charmap', b'\x8d', 0, 1, 'character maps to <undefined>')
    pass





