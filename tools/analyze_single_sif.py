import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools import process_files

def run():
    # Define UI elements and variables in a scope accessible to both columns
    col1, col2 = st.columns([1, 2])
    
    # Define all widgets and their variables at the top-level of the function
    # The `uploaded_file` variable is now guaranteed to be defined.
    with col1:
        st.header("Analyze Single SIF File")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"])
        st.markdown("""
        ┌─┬─┐<br>
        │ 1 │ 2 │<br>
        ├─┼─┤<br>
        │ 3 │ 4 │<br>
        └─┴─┘
        """, unsafe_allow_html=True)
        signal = st.selectbox("Signal", options=["UCNP", "dye"])
        show_fits = st.checkbox("Show fits")
        plot_brightness_histogram = st.checkbox("Plot brightness histogram")
        normalization = st.checkbox("Log Image Scaling")

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False
        
        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
        
        # This line will now work correctly because `uploaded_files` is always defined.
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region)
                
                if len(uploaded_files) > 1:
                    file_options = [f.name for f in uploaded_files]
                    selected_file_name = st.selectbox("Select a file to display:", options=file_options)
                else:
                    selected_file_name = uploaded_files[0].name
                
                if selected_file_name in processed_data:
                    data_to_plot = processed_data[selected_file_name]
                    df_selected = data_to_plot["df"]
                    image_data_cps = data_to_plot["image"]
                    
                    plot_col1, plot_col2 = st.columns(2)
                    
                    with plot_col1:
                        normalization_to_use = LogNorm() if normalization else None
                        fig_image = plot_brightness(image_data_cps, df_selected, show_fits=show_fits,
                                                     normalization=normalization_to_use, pix_size_um=0.1)
                        st.pyplot(fig_image)
                        
                        # Download button for the brightness plot
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
                                fig_hist = plot_histogram(combined_df)
                                st.pyplot(fig_hist)
                                
                                # Download button for the histogram
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
                            st.error(f"Data for file '{selected_file_name}' not found.")
            
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
