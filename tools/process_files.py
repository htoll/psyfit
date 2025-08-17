import streamlit as st
import os
import tempfile 
import pandas as pd
from matplotlib.colors import LogNorm
import io
from utils import integrate_sif, plot_brightness, plot_histogram 


@st.cache_data
def process_files(files_or_paths, region):
    """
    Accepts either:
      - iterable of Streamlit UploadedFile objects, or
      - iterable of string file paths

    Returns:
      processed_data: dict keyed by base filename
      combined_df: pandas DataFrame
    """
    temp_dir = tempfile.mkdtemp()
    local_paths = []  # [(display_name, local_path)]

    for item in files_or_paths:
        # Case 1: Streamlit UploadedFile-like object
        if hasattr(item, "name") and hasattr(item, "getbuffer"):
            display_name = os.path.basename(item.name)
            local_path = os.path.join(temp_dir, display_name)
            with open(local_path, "wb") as f:
                f.write(item.getbuffer())
            local_paths.append((display_name, local_path))

        # Case 2: Already a path string
        elif isinstance(item, str):
            display_name = os.path.basename(item)
            # Copy to a working temp dir (optional but keeps behavior consistent)
            local_path = os.path.join(temp_dir, display_name)
            if os.path.abspath(item) != os.path.abspath(local_path):
                shutil.copy2(item, local_path)
            local_paths.append((display_name, local_path))

        else:
            raise TypeError(
                f"Unsupported item type {type(item)} in process_files; "
                "expected UploadedFile or str path."
            )

    # ----- Your existing heavy processing goes here -----
    processed_data = {}
    combined_df = pd.DataFrame()

    for display_name, local_path in local_paths:
        # parse/process the .sif at `local_path`
        # populate `processed_data[display_name] = {"df": ..., "image": ...}`
        # and extend/concat into `combined_df`
        pass  # <-- keep your current logic here

    return processed_data, combined_df
