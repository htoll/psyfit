import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

def run():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Analyze SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=2)
        diagram = ("""
        ┌─┬─┐<br>
        │ 1 │ 2 │<br>
        ├─┼─┤<br>
        │ 3 │ 4 │<br>
        └─┴─┘
        """)#, unsafe_allow_html=True)
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help = diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help= "Changes detection method, UCNP for high SNR (sklearn peakfinder) while dye for low SNR (sklearn blob detection)")
        cmap = st.selectbox("Colormap", options = ["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])
        show_fits = st.checkbox("Show fits")
        plot_brightness_histogram = st.checkbox("Plot brightness histogram")
        normalization = st.checkbox("Log Image Scaling")

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
        
        # All dynamic output logic is now within this `with col2:` block
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region)

                # Now create the sub-columns for the plots below the selectbox
                plot_col1, plot_col2 = st.columns(2)

                with plot_col1:
                    if len(uploaded_files) > 1:
                        file_options = [f.name for f in uploaded_files]
                        selected_file_name = st.selectbox("Select sif to display:", options=file_options)
                    if selected_file_name in processed_data:
                        data_to_plot = processed_data[selected_file_name]
                        df_selected = data_to_plot["df"]
                        image_data_cps = data_to_plot["image"]
                    normalization_to_use = LogNorm() if normalization else None
                    fig_image = plot_brightness(
                        image_data_cps,
                        df_selected,
                        show_fits=show_fits,
                        normalization=normalization_to_use,
                        pix_size_um=0.1,
                        cmap = cmap
                    )
                    st.pyplot(fig_image)

                    svg_buffer = io.StringIO()
                    fig_image.savefig(svg_buffer, format='svg')
                    svg_data = svg_buffer.getvalue()
                    svg_buffer.close()
                    st.download_button(
                        label="Download PSFs",
                        data=svg_data,
                        file_name=f"{selected_file_name}.svg",
                        mime="image/svg+xml"
                    )
                    
                if plot_brightness_histogram and not combined_df.empty:
                    with plot_col2:                
                        # Auto-detect full data range
                        brightness_vals = combined_df['brightness_fit'].values
                        default_min = float(np.min(brightness_vals))
                        default_max = float(np.max(brightness_vals))
                
                        # User inputs
                        user_min = st.number_input("Min Brightness (pps)", value=default_min)
                        user_max = st.number_input("Max Brightness (pps)", value=default_max)

                        #binning
                        num_bins = st.number_input("# Bins:", value = 20)
                
                        if user_min < user_max:
                            fig_hist = plot_histogram(combined_df, min_val=user_min, max_val=user_max, num_bins = num_bins)
                            st.pyplot(fig_hist)
                
                            svg_buffer_hist = io.StringIO()
                            fig_hist.savefig(svg_buffer_hist, format='svg')
                            svg_data_hist = svg_buffer_hist.getvalue()
                            svg_buffer_hist.close()
                
                            st.download_button(
                                label="Download histogram",
                                data=svg_data_hist,
                                file_name="combined_histogram.svg",
                                mime="image/svg+xml"
                            )

                        
                        else:
                            st.warning("Min greater than max")

                else:
                    st.error(f"Data for file '{selected_file_name}' not found.")
            
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
