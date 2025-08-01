import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools import analyze_single_sif


def run():
    st.header("Analyze multiple SIF files")
    uploaded_files = st.file_uploader(
        "Upload .sif files",
        type="sif",
        accept_multiple_files=True
    )
    
  if uploaded_zip is not None:
      temp_dir = "temp_sif_batch"
      os.makedirs(temp_dir, exist_ok=True)

      # Unzip files
      with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
          zip_ref.extractall(temp_dir)

      # Find all .sif files
      sif_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".sif")])
      if not sif_files:
          st.warning("No .sif files found in uploaded ZIP.")
          return

      all_dfs = []
      brightness_column = "brightness"  # adjust to match your real column

      st.write(f"Found {len(sif_files)} .sif files.")
      for filename in sif_files:
          path = os.path.join(temp_dir, filename)
          try:
              df, _ = integrate_sif(path)
              df["source_file"] = filename
              all_dfs.append(df)
          except Exception as e:
              st.error(f"Error processing {filename}: {e}")

      if not all_dfs:
          st.warning("No data could be extracted.")
          return

      # Combine all data
      full_df = pd.concat(all_dfs, ignore_index=True)

      # Histogram of brightness
      fig_hist = plot_histogram(full_df)

      st.pyplot(fig_hist)

      # Download buttons
      svg_buffer = io.StringIO()
      fig_hist.savefig(svg_buffer, format="svg")
      svg_data = svg_buffer.getvalue()
      svg_buffer.close()

      st.download_button(
          label="Download Brightness Histogram (SVG)",
          data=svg_data,
          file_name="brightness_histogram.svg",
          mime="image/svg+xml"
      )

      csv_data = full_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="Download Data (CSV)",
          data=csv_data,
          file_name="brightness_data.csv",
          mime="text/csv"
      )

      # Dropdown to view a single file
      selected_file = st.selectbox("Preview a specific SIF file", sif_files)
      selected_path = os.path.join(temp_dir, selected_file)
      try:
          df_single, image_data_cps = integrate_sif(selected_path)
          fig_image = plot_brightness(image_data_cps, df_single)
          st.pyplot(fig_image)
      except Exception as e:
          st.error(f"Error loading preview file: {e}")
