import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools import process_files

def run():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Analyze Single SIF File")
        uploaded_file = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
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
        if "show_data_clicked" not in st.session_state:
            st.session_state.show_data_clicked = False

        if st.button("Analyze"):
            st.session_state.show_data_clicked = True

        if uploaded_files and st.session_state.get("run_analysis", False):
            processed_data, combined_df = process_files(uploaded_files, region)
            if len(uploaded_files) > 1:
                file_options = [f.name for f in uploaded_files]
                selected_file_name = st.selectbox("Select a file to display:", options=file_options)
            else:
                selected_file_name = uploaded_files[0].name
        if selected_file_name in processed_data:
                data_to_plot = processed_data[selected_file_name]
                df = data_to_plot["df"]
                image_data_cps = data_to_plot["image"]
                
                plot_col1, plot_col2 = st.columns(2)
                
                with plot_col1:
                    # Use the selected file's data for the brightness plot
                    normalization_to_use = LogNorm() if normalization else None
                    fig_image = plot_brightness(image_data_cps, df, show_fits=show_fits,
                                                 normalization=normalization_to_use, pix_size_um=0.1)
                    st.pyplot(fig_image)
                    
                    # Add download button for the brightness image
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
                        # Use the combined dataframe for the histogram plot
                        fig_hist = plot_histogram(combined_df)
                        st.pyplot(fig_hist)
                        
                        # Add download button for the histogram
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
                st.warning("Please upload a .sif file.")


        # if st.session_state.show_data_clicked:
        #     if uploaded_file is not None:
        #         try:
        #             os.makedirs("temp", exist_ok=True)
        #             file_path = os.path.join("temp", uploaded_file.name)
        #             with open(file_path, "wb") as f:
        #                 f.write(uploaded_file.getbuffer())
        #             df, image_data_cps = integrate_sif(file_path, region=region)

        #             plot_col1, plot_col2 = st.columns(2)

        #             with plot_col1:
        #                 fig_image = plot_brightness(image_data_cps, df, show_fits=show_fits,
        #                                             normalization=normalization, pix_size_um=0.1)
        #                 st.pyplot(fig_image)
        #                 svg_buffer = io.StringIO()
        #                 fig_image.savefig(svg_buffer, format='svg')
        #                 svg_data = svg_buffer.getvalue()
        #                 svg_buffer.close()

        #                 st.download_button(
        #                     label="Download PSFs",
        #                     data=svg_data,
        #                     file_name=f"{file_path[5:]}.svg",
        #                     mime="image/svg+xml"
        #                 )

        #             if plot_brightness_histogram:
        #                 with plot_col2:
        #                     fig_hist = plot_histogram(df)
        #                     st.pyplot(fig_hist)
        #                     svg_buffer = io.StringIO()
        #                     fig_hist.savefig(svg_buffer, format='svg')
        #                     svg_data = svg_buffer.getvalue()
        #                     svg_buffer.close()

        #                     st.download_button(
        #                         label="Download histogram",
        #                         data=svg_data,
        #                         file_name=f"{file_path}.svg",
        #                         mime="image/svg+xml"
        #                     )

        #         except Exception as e:
        #             st.error(f"Error processing file: {e}")
        #     else:
        #         st.warning("Please upload a .sif file.")
