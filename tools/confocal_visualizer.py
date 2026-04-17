import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import re
import tifffile

# Import the existing dat reader from your tools
from tools.confocal_brightness import read_dat_image

def get_custom_lut(name):
    """Generates custom colormaps or returns standard matplotlib cmaps."""
    standard_cmaps = ['magma', 'viridis', 'inferno', 'plasma', 'hot', 'gray', 'bone', 'ocean']
    if name in standard_cmaps:
        return plt.get_cmap(name)

    zero_channel = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    full_channel = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    
    luts = {
        "Greyscale": {'red': full_channel, 'green': full_channel, 'blue': full_channel},
        "Red hot":   {'red': full_channel, 'green': zero_channel, 'blue': zero_channel},
        "Green hot": {'red': zero_channel, 'green': full_channel, 'blue': zero_channel},
        "Cyan hot":  {'red': zero_channel, 'green': full_channel, 'blue': full_channel},
        "Blue":      {'red': zero_channel, 'green': zero_channel, 'blue': full_channel},
        "Green":     {'red': zero_channel, 'green': full_channel, 'blue': zero_channel},
        "Pink":      {'red': full_channel, 'green': zero_channel, 'blue': full_channel}
    }
    
    if name in luts:
        return LinearSegmentedColormap(name, luts[name])
    return plt.get_cmap('gray') # Default

def get_scale_factor(unit_str):
    """Converts common metric strings to a multiplier for nanometers."""
    if not unit_str: return 1.0
    u = unit_str.lower()
    if u in ['nm']: return 1.0
    if u in ['um', 'µm', 'u']: return 1000.0
    if u in ['mm']: return 1000000.0
    if u in ['cm']: return 10000000.0
    if u in ['m']: return 1000000000.0
    return 1.0

def parse_filename(filename, conditions_dict, px_tag, dwell_tag, acc_tag):
    """Extracts properties and condition from the filename, normalizing to nm and seconds."""
    params = {'pixel_size_nm': None, 'dwell_time_s': None, 'accumulation': 1.0, 'condition': "Unknown"}
    
    m_pix = re.search(r"(\d+(?:\.\d+)?)\s*(nm|um|µm|mm|cm)?\s*" + re.escape(px_tag), filename, re.IGNORECASE)
    if m_pix: 
        params['pixel_size_nm'] = float(m_pix.group(1)) * get_scale_factor(m_pix.group(2))
        
    m_dwell = re.search(r"(\d+(?:\.\d+)?)\s*(ms|us|µs|s)?\s*" + re.escape(dwell_tag), filename, re.IGNORECASE)
    if m_dwell:
        val = float(m_dwell.group(1))
        unit = m_dwell.group(2)
        if unit == 's': params['dwell_time_s'] = val
        elif unit in ['us', 'µs']: params['dwell_time_s'] = val / 1e6
        else: params['dwell_time_s'] = val / 1000.0 
        
    m_acc = re.search(r"(\d+(?:\.\d+)?)\s*" + re.escape(acc_tag), filename, re.IGNORECASE)
    if m_acc: params['accumulation'] = float(m_acc.group(1))
        
    for search_term, display_name in conditions_dict.items():
        if re.search(re.escape(search_term), filename, re.IGNORECASE):
            params['condition'] = display_name
            params['condition_search'] = search_term # retain original mapping for reference
            break
            
    return params

def extract_tiff_metadata(file_buffer):
    """Extracts metadata from TIFF tags, converting native units to nm for direct comparison."""
    meta = {'pixel_size_nm': None, 'dwell_time_s': None, 'accumulation': None}
    try:
        file_buffer.seek(0)
        with tifffile.TiffFile(file_buffer) as tif:
            if not tif.pages: return meta
            page = tif.pages[0]
            
            desc = page.tags.get('ImageDescription')
            if desc:
                desc_str = desc.value.lower() if isinstance(desc.value, str) else ""
                
                m_px = re.search(r'pixel\s*size.*?(\d+\.\d+|\d+e[-+]?\d+|\d+)\s*(nm|um|µm|mm|cm)?', desc_str)
                if m_px: meta['pixel_size_nm'] = float(m_px.group(1)) * get_scale_factor(m_px.group(2))
                
                m_dwell = re.search(r'dwell.*?(\d+\.\d+|\d+e[-+]?\d+|\d+)\s*(ms|us|µs|s)?', desc_str)
                if m_dwell:
                    val = float(m_dwell.group(1))
                    unit = m_dwell.group(2)
                    if unit == 's': meta['dwell_time_s'] = val
                    elif unit in ['us', 'µs']: meta['dwell_time_s'] = val / 1e6
                    else: meta['dwell_time_s'] = val / 1000.0 

                m_acc = re.search(r'accum.*?(\d+\.\d+|\d+)', desc_str)
                if m_acc: meta['accumulation'] = float(m_acc.group(1))
                
            if meta['pixel_size_nm'] is None:
                res_x = page.tags.get('XResolution')
                res_unit = page.tags.get('ResolutionUnit')
                if res_x and res_x.value[0] > 0:
                    size_in_units = res_x.value[1] / res_x.value[0] 
                    u_val = res_unit.value if res_unit else 2 
                    if u_val == 3: meta['pixel_size_nm'] = size_in_units * 1e7
                    elif u_val == 2: meta['pixel_size_nm'] = size_in_units * 2.54e7
                        
    except Exception:
        pass
    return meta

def blend_images(img1_rgba, img2_rgba):
    """Additively blends two RGBA images."""
    blended = img1_rgba + img2_rgba
    blended[:, :, :3] = np.clip(blended[:, :, :3], 0, 1) 
    blended[:, :, 3] = 1.0 
    return blended

def add_scale_bar(ax, shape, pixel_size_nm):
    """Draws a cleanly rounded scale bar on the image encapsulated in a grey box."""
    if not pixel_size_nm or pixel_size_nm <= 0: return
    h, w = shape
    total_width_nm = w * pixel_size_nm
    
    # Target scale bar size: 20% of the image width
    target_nm = total_width_nm * 0.20 
    if target_nm <= 0: return

    magnitude = 10 ** np.floor(np.log10(target_nm))
    val = target_nm / magnitude
    if val < 2: nice = 1
    elif val < 5: nice = 2
    else: nice = 5
    bar_length_nm = nice * magnitude
    
    bar_px = bar_length_nm / pixel_size_nm
    
    if bar_length_nm >= 1000:
        label = f"{int(bar_length_nm/1000) if bar_length_nm%1000==0 else bar_length_nm/1000} µm"
    else:
        label = f"{int(bar_length_nm)} nm"
        
    x_right = w * 0.95
    x_left = x_right - bar_px
    y_pos = h * 0.05 
    
    # Draw semi-transparent background box covering line and text
    box_h = h * 0.08
    box_w = bar_px + w * 0.04
    rect = plt.Rectangle((x_left - w * 0.02, y_pos - h * 0.015), box_w, box_h, 
                         color='black', alpha=0.5, zorder=1, lw=0)
    ax.add_patch(rect)
    
    # Draw line
    ax.plot([x_left, x_right], [y_pos + h*0.015, y_pos + h*0.015], color='white', lw=3, solid_capstyle='butt', zorder=2)
    
    # Draw Text centered above line
    ax.text((x_left + x_right)/2, y_pos + h*0.035, label, color='white', 
            ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

def get_norm(img_array, min_pct, max_pct, log_scale):
    """Generates the normalization bounds based on percentiles."""
    vmin = np.percentile(img_array, min_pct)
    vmax = np.percentile(img_array, max_pct)
    
    if vmin >= vmax:
        vmax = vmin + 1e-6
        
    if log_scale:
        vmin_log = max(vmin, 1e-3)
        vmax_log = max(vmax, 1e-2)
        if vmin_log >= vmax_log: vmax_log = vmin_log * 10
        return LogNorm(vmin=vmin_log, vmax=vmax_log)
    else:
        return Normalize(vmin=vmin, vmax=vmax)

def run():
    st.header("Confocal Visualization & Merge Tool")
    
    with st.sidebar:
        st.subheader("Data Import")
        uploaded_files = st.file_uploader("Upload .tif or .dat files", type=["tif", "tiff", "dat"], accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("Filename Parsing Tags")
        px_tag = st.text_input("Pixel Size Tag", value="nmpx")
        dwell_tag = st.text_input("Dwell Time Tag", value="msDwell")
        acc_tag = st.text_input("Accumulation Tag", value="lineaccum")
        
        st.markdown("---")
        st.subheader("Conditions & Adjustments")
        cond_input = st.text_area("Known Conditions (comma separated)", value="Erprobe, 561:AF561, DAPI",
                                  help="Format: 'SearchString' OR 'SearchString:DisplayName'. E.g. '561:AF561' searches for 561 but plots as AF561.")
        
        # Parse Known conditions + mapping aliases
        conditions_dict = {}
        display_names = []
        for c in cond_input.split(","):
            c = c.strip()
            if not c: continue
            if ":" in c:
                search, display = c.split(":", 1)
                search, display = search.strip(), display.strip()
                conditions_dict[search] = display
                display_names.append(display)
            else:
                conditions_dict[c] = c
                display_names.append(c)
        
        condition_settings = {}
        lut_options = ["Greyscale", "Cyan hot", "Red hot", "Green hot", "Blue", "Green", "Pink", 
                       "magma", "viridis", "inferno", "plasma", "hot", "gray", "bone", "ocean"]
        
        # 5-Column setup
        for i, cond in enumerate(display_names):
            st.markdown(f"**{cond}**")
            c1, c2, c3, c4, c5 = st.columns([2.0, 0.8, 1.0, 1.0, 1.0])
            with c1:
                lut = st.selectbox(f"LUT", lut_options, key=f"lut_{cond}", index=i % len(lut_options), label_visibility="collapsed")
            with c2:
                order = st.number_input(f"Ord", value=i+1, step=1, key=f"order_{cond}", label_visibility="collapsed", help="Plotting order")
            with c3:
                min_p = st.number_input("Min%", value=1.0, min_value=0.0, max_value=100.0, key=f"min_{cond}", label_visibility="collapsed", help="Min Contrast %")
            with c4:
                max_p = st.number_input("Max%", value=99.9, min_value=0.0, max_value=100.0, key=f"max_{cond}", label_visibility="collapsed", help="Max Contrast %")
            with c5:
                no_log = st.checkbox("Lin", key=f"nolog_{cond}", help="Force linear scale (ignore Log toggle)")
                
            condition_settings[cond] = {'lut': lut, 'order': order, 'min_p': min_p, 'max_p': max_p, 'no_log': no_log}
            st.write("") 

        st.markdown("---")
        st.subheader("Display Settings")
        display_mode = st.radio("Intensity Units", ["Photons per Second (pps)", "Raw Pixel Values"])
        log_scale = st.toggle("Log Scale Images (Global)", value=False)
        grid_plot = st.toggle("Create Merge Grid Plot")

    if not uploaded_files:
        st.info("Upload files to begin.")
        return

    processed_data = []
    unique_conditions = set()

    for file in uploaded_files:
        filename = file.name
        title_params = parse_filename(filename, conditions_dict, px_tag, dwell_tag, acc_tag)
        
        image_data = None
        meta_params = {'pixel_size_nm': None, 'dwell_time_s': None, 'accumulation': None}
        
        if filename.lower().endswith(('.tif', '.tiff')):
            try:
                file.seek(0)
                image_data = tifffile.imread(file)
                meta_params = extract_tiff_metadata(file)
            except Exception as e:
                st.error(f"Failed to read TIFF {filename}: {e}")
                continue
        elif filename.lower().endswith('.dat'):
            image_data = read_dat_image(file)
            
        if image_data is None:
            continue

        final_params = title_params.copy()
        
        if meta_params['pixel_size_nm'] is not None:
            final_params['pixel_size_nm'] = meta_params['pixel_size_nm']
            if title_params['pixel_size_nm'] is not None:
                if abs(meta_params['pixel_size_nm'] - title_params['pixel_size_nm']) / max(meta_params['pixel_size_nm'], 1) > 0.01:
                    st.warning(f"**Mismatch in {filename}:** Pixel size is {title_params['pixel_size_nm']:.1f} nm in title, but {meta_params['pixel_size_nm']:.1f} nm in metadata. Defaulting to metadata.")

        if meta_params['dwell_time_s'] is not None:
            final_params['dwell_time_s'] = meta_params['dwell_time_s']
            if title_params['dwell_time_s'] is not None:
                if abs(meta_params['dwell_time_s'] - title_params['dwell_time_s']) > 1e-6:
                    st.warning(f"**Mismatch in {filename}:** Dwell is {title_params['dwell_time_s']}s in title, but {meta_params['dwell_time_s']}s in metadata. Defaulting to metadata.")

        if meta_params['accumulation'] is not None:
            final_params['accumulation'] = meta_params['accumulation']
            if title_params['accumulation'] != meta_params['accumulation']:
                st.warning(f"**Mismatch in {filename}:** Accumulation is {title_params['accumulation']} in title, but {meta_params['accumulation']} in metadata. Defaulting to metadata.")

        dwell_s = final_params['dwell_time_s'] if final_params['dwell_time_s'] else 1e-3
        acc = final_params['accumulation'] if final_params['accumulation'] else 1.0
        denom = dwell_s * acc
        
        pps_data = image_data / denom if denom > 0 else image_data
        unique_conditions.add(final_params['condition'])
        
        processed_data.append({
            'filename': filename,
            'condition': final_params['condition'],
            'raw': image_data,
            'pps': pps_data,
            'params': final_params
        })

    if not processed_data:
        return

    st.markdown("---")
    
    data_key = 'pps' if "Photons" in display_mode else 'raw'
    label = 'pps' if "Photons" in display_mode else 'intensity'

    if not grid_plot:
        st.subheader("Individual Images")
        processed_data = sorted(processed_data, key=lambda x: condition_settings.get(x['condition'], {}).get('order', 999))
        
        n_files = len(processed_data)
        n_cols = min(3, n_files)
        n_rows = int(np.ceil(n_files / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
            
        for i, data in enumerate(processed_data):
            ax = axes[i]
            img = data[data_key]
            cond_opts = condition_settings.get(data['condition'], {})
            
            cmap = get_custom_lut(cond_opts.get('lut', 'Greyscale'))
            use_log = log_scale and not cond_opts.get('no_log', False)
            norm = get_norm(img, cond_opts.get('min_p', 1.0), cond_opts.get('max_p', 99.9), use_log)
            
            im = ax.imshow(img, cmap=cmap, norm=norm, origin='lower')
            add_scale_bar(ax, img.shape, data['params'].get('pixel_size_nm'))
            
            # File info in bottom corner
            info_text = f"{data['filename']}\nMin: {cond_opts.get('min_p')}% | Max: {cond_opts.get('max_p')}%"
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, color='lightgrey', fontsize=6, ha='left', va='bottom', zorder=4)
            
            plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            ax.set_title(data['condition'], fontsize=12, fontweight='bold')
            ax.axis('off')
            
        for ax in axes[len(processed_data):]:
            ax.axis('off')
            
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.subheader("Merge Grid Plot")
        cond_list = sorted(list(unique_conditions), key=lambda x: condition_settings.get(x, {}).get('order', 999))
        n_conds = len(cond_list)
        
        rep_images = {}
        rep_raw_data = {}
        rep_params = {}
        
        for cond in cond_list:
            for d in processed_data:
                if d['condition'] == cond:
                    img = d[data_key]
                    rep_raw_data[cond] = img
                    rep_params[cond] = d['params']
                    
                    cond_opts = condition_settings.get(cond, {})
                    cmap = get_custom_lut(cond_opts.get('lut', 'Greyscale'))
                    use_log = log_scale and not cond_opts.get('no_log', False)
                    norm = get_norm(img, cond_opts.get('min_p', 1.0), cond_opts.get('max_p', 99.9), use_log)
                    
                    rgba_img = cmap(norm(img))
                    rep_images[cond] = rgba_img
                    break
        
        fig, axes = plt.subplots(n_conds, n_conds + 1, figsize=(4 * (n_conds + 0.5), 4 * n_conds), 
                                 gridspec_kw={'width_ratios': [1]*n_conds + [0.08]})
        if n_conds == 1:
            axes = np.array([[axes[0], axes[1]]])
            
        for row, cond_row in enumerate(cond_list):
            for col, cond_col in enumerate(cond_list):
                ax = axes[row, col]
                
                img_row = rep_images.get(cond_row)
                img_col = rep_images.get(cond_col)
                raw_shape = rep_raw_data.get(cond_row).shape if rep_raw_data.get(cond_row) is not None else None
                
                if img_row is None or img_col is None:
                    ax.axis('off')
                    continue
                
                if row == col:
                    ax.imshow(img_row, origin='lower')
                    ax.set_title(cond_row, fontsize=12, fontweight='bold')
                    
                    cond_opts = condition_settings.get(cond_row, {})
                    filename = rep_params[cond_row].get('filename', '')
                    info_text = f"{filename}\nMin: {cond_opts.get('min_p')}% | Max: {cond_opts.get('max_p')}%"
                    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, color='lightgrey', fontsize=6, ha='left', va='bottom', zorder=4)

                else:
                    blended = blend_images(img_row, img_col)
                    ax.imshow(blended, origin='lower')
                    ax.set_title(f"{cond_row} + {cond_col}", fontsize=10)
                    
                    cond_opts_row = condition_settings.get(cond_row, {})
                    cond_opts_col = condition_settings.get(cond_col, {})
                    info_text = (f"{cond_row} -> Min: {cond_opts_row.get('min_p')}% | Max: {cond_opts_row.get('max_p')}%\n"
                                 f"{cond_col} -> Min: {cond_opts_col.get('min_p')}% | Max: {cond_opts_col.get('max_p')}%")
                    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, color='lightgrey', fontsize=5, ha='left', va='bottom', zorder=4)
                
                # Apply scale bar to every panel
                if raw_shape:
                    add_scale_bar(ax, raw_shape, rep_params[cond_row].get('pixel_size_nm'))
                    
                ax.axis('off')
                
                if row == 0:
                    ax.text(0.5, 1.1, cond_col, transform=ax.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold')
                if col == 0:
                    ax.text(-0.1, 0.5, cond_row, transform=ax.transAxes, ha='right', va='center', fontsize=14, fontweight='bold', rotation=90)

            cbar_ax = axes[row, -1]
            img_row_raw = rep_raw_data.get(cond_row)
            
            if img_row_raw is not None:
                cond_opts = condition_settings.get(cond_row, {})
                cmap = get_custom_lut(cond_opts.get('lut', 'Greyscale'))
                use_log = log_scale and not cond_opts.get('no_log', False)
                norm = get_norm(img_row_raw, cond_opts.get('min_p', 1.0), cond_opts.get('max_p', 99.9), use_log)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, cax=cbar_ax, label=label)
            else:
                cbar_ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    run()
