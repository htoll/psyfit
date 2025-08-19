import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

def run():
    col1, col2 = st.columns([1, 2])
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=2, help = '''
        Stringency of fit, higher value is more selective:  
        -UCNP signal sets absolute peak cut off  
        -Dye signal sets sensitivity of blob detection
        ''')        
        diagram = """ Splits sif into quadrants (256x256 px):  
        ┌─┬─┐  
        │ 1 │ 2 │  
        ├─┼─┤  
        │ 3 │ 4 │  
        └─┴─┘
        """
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help = diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help= '''Changes detection method:  
                                                                - UCNP for high SNR (sklearn peakfinder)  
                                                                - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options = ["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])


    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False
    
        plot_col1, plot_col2 = st.columns(2)
    
        with plot_col1:
            show_fits = st.checkbox("Show fits")
            plot_brightness_histogram = True
            normalization = st.checkbox("Log Image Scaling")
    
            if st.button("Analyze"):
                st.session_state.analyze_clicked = True
    
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region, threshold = threshold, signal=signal)
    
                if len(uploaded_files) > 1:
                    file_options = [f.name for f in uploaded_files]
                    selected_file_name = st.selectbox("Select sif to display:", options=file_options)
                else:
                    selected_file_name = uploaded_files[0].name
    
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
                        cmap=cmap
                    )
    
                    with plot_col1:
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
                        if combined_df is not None and not combined_df.empty:
                            csv_bytes = df_to_csv_bytes(combined_df)
                            st.download_button(
                                label="Download as CSV",
                                data=csv_bytes,
                                file_name=f"{os.path.splitext(selected_file_name)[0]}_compiled.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No compiled data available to download yet.")
    
                    with plot_col2:
                        if plot_brightness_histogram and not combined_df.empty:
                            brightness_vals = combined_df['brightness_fit'].values
                            default_min_val = float(np.min(brightness_vals))
                            default_max_val = float(np.max(brightness_vals))
    
                            user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                            user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")
    
                            try:
                                user_min = float(user_min_val_str)
                                user_max = float(user_max_val_str)
                            except ValueError:
                                st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                                return
    
                            num_bins = st.number_input("# Bins:", value=50)
    
                            if user_min < user_max:
                                fig_hist, _, _ = plot_histogram(
                                    combined_df,
                                    min_val=user_min,
                                    max_val=user_max,
                                    num_bins=num_bins
                                )
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
                                st.warning("Min greater than max.")
                else:
                    st.error(f"Data for file '{selected_file_name}' not found.")
    
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
    

