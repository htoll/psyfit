import streamlit as st
import numpy as np
import cv2
import tifffile
import io
import contextlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree

# ==============================================================================
#  HELPER FUNCTIONS (UNCHANGED FROM PREVIOUS VERSION)
# ==============================================================================
def findOptimalPeaks(image, minPeaks=10, maxPeaks=100, minRoundness=0.8):
    # This function is unchanged.
    print(f"Searching for optimal threshold in image of shape {image.shape}...")
    image_max = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else np.max(image)
    start_thresh, end_thresh = int(image_max * 0.98), int(image_max * 0.20)
    step = -max(1, int(image_max / 200))

    for threshold in range(start_thresh, end_thresh, step):
        binaryMask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        goodPeaks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity >= minRoundness:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    goodPeaks.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
        if minPeaks <= len(goodPeaks) <= maxPeaks:
            print(f"  Success! Found optimal threshold: {threshold} with {len(goodPeaks)} peaks.")
            return np.array(goodPeaks), threshold
    print("  Warning: Could not find an optimal threshold. Returning empty set.")
    return np.array([]), -1

def matchFeatures(imageSource, pointsSource, imageTarget, pointsTarget, patchSize=15, searchRadius=75):
    # This function is unchanged.
    halfPatch = patchSize // 2
    matchedSourcePoints, matchedTargetPoints = [], []
    sourcePadded = np.pad(imageSource, halfPatch, mode='constant')
    targetPadded = np.pad(imageTarget, halfPatch, mode='constant')
    for pS in pointsSource:
        pS_padded = (pS + halfPatch).astype(int)
        sourcePatch = sourcePadded[pS_padded[1]-halfPatch : pS_padded[1]+halfPatch+1, pS_padded[0]-halfPatch : pS_padded[0]+halfPatch+1]
        bestMatchPoint, minSsd = None, float('inf')
        x, y = pS.astype(int)
        yMin, yMax = max(0, y - searchRadius), min(imageTarget.shape[0], y + searchRadius)
        xMin, xMax = max(0, x - searchRadius), min(imageTarget.shape[1], x + searchRadius)
        candidatePoints = [pT for pT in pointsTarget if xMin <= pT[0] < xMax and yMin <= pT[1] < yMax]
        for pT in candidatePoints:
            pT_padded = (pT + halfPatch).astype(int)
            targetPatch = targetPadded[pT_padded[1]-halfPatch : pT_padded[1]+halfPatch+1, pT_padded[0]-halfPatch : pT_padded[0]+halfPatch+1]
            if sourcePatch.shape != targetPatch.shape: continue
            ssd = np.sum((sourcePatch.astype(np.float32) - targetPatch.astype(np.float32))**2)
            if ssd < minSsd:
                minSsd, bestMatchPoint = ssd, pT
        if bestMatchPoint is not None:
            matchedSourcePoints.append(pS)
            matchedTargetPoints.append(bestMatchPoint)
    return np.array(matchedSourcePoints), np.array(matchedTargetPoints)

def warpImagePiecewise(sourceImage, sourcePoints, targetPoints):
    # This function is unchanged.
    targetTri = Delaunay(targetPoints)
    warpedImage = np.zeros(sourceImage.shape, dtype=sourceImage.dtype)
    for simplex in targetTri.simplices:
        srcTriangle = sourcePoints[simplex].astype(np.float32)
        tgtTriangle = targetPoints[simplex].astype(np.float32)
        (x, y, w, h) = cv2.boundingRect(tgtTriangle)
        if w == 0 or h == 0: continue
        tgtCropped = tgtTriangle - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tgtCropped.astype(np.int32), 255)
        transformMatrix = cv2.getAffineTransform(srcTriangle, tgtTriangle)
        warpedRegion = cv2.warpAffine(sourceImage, transformMatrix, (sourceImage.shape[1], sourceImage.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        roi = warpedImage[y:y+h, x:x+w]
        roi[mask > 0] = warpedRegion[y:y+h, x:x+w][mask > 0]
    return warpedImage

def getQuadrant(image, quad_index):
    # This function is unchanged.
    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2
    quadrants = {1: image[0:h_half, 0:w_half], 2: image[0:h_half, w_half:w], 3: image[h_half:h, 0:w_half], 4: image[h_half:h, w_half:w]}
    return quadrants.get(quad_index)

def getQuadrantOffset(image_shape, quad_index):
    # This function is unchanged.
    h, w = image_shape[:2]
    h_half, w_half = h // 2, w // 2
    offsets = {1: (0, 0), 2: (w_half, 0), 3: (0, h_half), 4: (w_half, h_half)}
    return offsets.get(quad_index, (0, 0))

def find_pairwise_matches(points1, points2, radius):
    # This function is unchanged.
    if len(points1) == 0 or len(points2) == 0:
        return np.array([]), np.array([])
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, idx1 = tree1.query(points2, k=1)
    dist2, idx2 = tree2.query(points1, k=1)
    matches1 = np.arange(len(points2))
    mutual_mask = (idx2[idx1[matches1]] == matches1) & (dist1 < radius)
    matched_indices1 = idx1[matches1[mutual_mask]]
    matched_indices2 = matches1[mutual_mask]
    return points1[matched_indices1], points2[matched_indices2]

# ==============================================================================
#  MAIN REGISTRATION WORKFLOW (MODIFIED TO RETURN INITIAL PEAKS)
# ==============================================================================
def run_registration_workflow(imgRegistration, imgData, activeQuads, peakParams, matchParams, correlationRadius):
    """Encapsulates the entire registration process using a robust pairwise approach."""
    # --- Part 1: Initial Peak Finding ---
    print("\nStep 1: Finding initial peaks in all active quadrants...")
    initial_peaks = {}
    for quad_idx in activeQuads:
        quad_img = getQuadrant(imgRegistration, quad_idx)
        peaks, _ = findOptimalPeaks(quad_img, **peakParams)
        if len(peaks) > 0:
            offset = getQuadrantOffset(imgRegistration.shape, quad_idx)
            initial_peaks[quad_idx] = peaks + offset
            print(f"  Quadrant {quad_idx}: Found {len(initial_peaks[quad_idx])} peaks.")
        else:
            initial_peaks[quad_idx] = np.array([])
            print(f"  Quadrant {quad_idx}: No peaks found.")
    
    # --- Part 2: Pairwise Registration Loop ---
    transformations = {}
    if not activeQuads:
        print("Warning: No active quadrants selected.")
        return np.zeros((4, imgData.shape[0]//2, imgData.shape[1]//2), dtype=imgData.dtype), initial_peaks

    refQuadIndex = activeQuads[0]
    refPeaksGlobal = initial_peaks.get(refQuadIndex, np.array([]))
    
    if len(refPeaksGlobal) < 3:
        print(f"Error: Cannot perform registration. Reference quadrant {refQuadIndex} has fewer than 3 peaks.")
    else:
        sourceQuadIndices = [q for q in activeQuads if q != refQuadIndex]
        for srcQuadIndex in sourceQuadIndices:
            print(f"\n--- Processing Pair: Reference {refQuadIndex} <-> Source {srcQuadIndex} ---")
            srcPeaksGlobal = initial_peaks.get(srcQuadIndex, np.array([]))
            if len(srcPeaksGlobal) < 3:
                print(f"Skipping quadrant {srcQuadIndex}: fewer than 3 peaks found."); continue
            
            print(f"Step 2a: Correlating peaks within {correlationRadius} pixel radius...")
            corr_ref, corr_src = find_pairwise_matches(refPeaksGlobal, srcPeaksGlobal, correlationRadius)
            print(f"  Found {len(corr_ref)} geometrically correlated pairs.")
            if len(corr_ref) < 3:
                print(f"Skipping quadrant {srcQuadIndex}: fewer than 3 correlated peaks."); continue

            print("Step 2b: Matching feature patches for final refinement...")
            final_src_pts, final_ref_pts = matchFeatures(imgRegistration, corr_src, imgRegistration, corr_ref, **matchParams)
            print(f"  Found {len(final_src_pts)} final matching points after patch comparison.")
            if len(final_src_pts) >= 3:
                transformations[srcQuadIndex] = (final_src_pts, final_ref_pts)
            else:
                print(f"Skipping quadrant {srcQuadIndex}: fewer than 3 final matched points.")

    # --- Part 3: Build Final Output Stack ---
    print("\nStep 3: Assembling final 4-channel image...")
    quad_h, quad_w = imgData.shape[0] // 2, imgData.shape[1] // 2
    output_stack = np.zeros((4, quad_h, quad_w), dtype=imgData.dtype)
    for i in range(1, 5):
        if i in activeQuads:
            if i in transformations:
                srcPoints, tgtPoints = transformations[i]
                warpedFullDataImage = warpImagePiecewise(imgData, srcPoints, tgtPoints)
                ref_x, ref_y = getQuadrantOffset(imgData.shape, refQuadIndex)
                output_stack[i - 1] = warpedFullDataImage[ref_y:ref_y+quad_h, ref_x:ref_x+quad_w]
                print(f"  Quadrant {i}: Placed registered data into Channel {i-1}.")
            else:
                output_stack[i - 1] = getQuadrant(imgData, i)
                if i != refQuadIndex:
                    print(f"  Quadrant {i}: Could not be aligned. Using original data for Channel {i-1}.")
                else:
                    print(f"  Quadrant {i}: (Reference) Using original data for Channel {i-1}.")
    
    print("\nProcessing complete.")
    # --- CHANGE: RETURN THE INITIAL PEAKS DICTIONARY FOR VISUALIZATION ---
    return output_stack, initial_peaks

# ==============================================================================
#  STREAMLIT GUI (PLOTTING LOGIC IS MODIFIED)
# ==============================================================================
def run():
    st.set_page_config(layout="wide")
    st.title("🔬 Quadrant-Based Image Cross-Registration")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Registration Setup")
        f_reg = st.file_uploader("Upload Registration Image (with fiducials)", type=["tif", "tiff"])
        with st.expander("Set Registration Parameters", expanded=True):
            activeQuads = st.multiselect("Active quadrants (first is reference)", options=[1, 2, 3, 4], default=[1, 2, 3, 4])
            st.markdown("###### Peak Detection")
            minPeaks = st.slider("Min desired peaks (per quad)", 5, 500, 10, key="min_peaks")
            maxPeaks = st.slider("Max desired peaks (per quad)", 10, 1000, 200, key="max_peaks")
            minRoundness = st.slider("Min peak roundness", 0.0, 1.0, 0.75, 0.05, key="roundness")
            st.markdown("###### Geometric Correlation")
            correlationRadius = st.slider("Peak correlation radius (pixels)", 1, 20, 4, key="correlation_radius")
            st.markdown("###### Feature Matching")
            patchSize = st.slider("Match patch size", 5, 51, 21, step=2, key="patch_size")
            searchRadius = st.slider("Match search radius (pixels)", 10, 500, 100, key="search_radius")
    with col2:
        st.subheader("2. Data Setup")
        f_data = st.file_uploader("Upload Data Image (to be warped)", type=["tif", "tiff"])
    st.markdown("---")
    if f_reg and f_data:
        if st.button("🚀 Run Registration", use_container_width=True):
            imgRegistration = tifffile.imread(io.BytesIO(f_reg.read()))
            imgData = tifffile.imread(io.BytesIO(f_data.read()))
            peakParams = {'minPeaks': minPeaks, 'maxPeaks': maxPeaks, 'minRoundness': minRoundness}
            matchParams = {'patchSize': patchSize, 'searchRadius': searchRadius}
            log_stream = io.StringIO()
            with st.spinner("Registration in progress... this may take a moment."):
                with contextlib.redirect_stdout(log_stream):
                    output_stack, detected_peaks_dict = run_registration_workflow(
                        imgRegistration, imgData, activeQuads, peakParams, matchParams, correlationRadius
                    )
            st.session_state['output_stack'] = output_stack
            st.session_state['log'] = log_stream.getvalue()
            # --- CHANGE: STORE THE DICTIONARY OF PEAKS ---
            st.session_state['detected_peaks'] = detected_peaks_dict
            st.session_state['registration_image'] = imgRegistration
            
    if 'output_stack' in st.session_state and st.session_state['output_stack'] is not None:
        st.success("✅ Registration complete!")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            # --- CHANGE: NEW PLOTTING LOGIC FOR ALL INITIAL PEAKS ---
            st.subheader("Initially Detected Peaks")
            fig_peaks, ax_peaks = plt.subplots(figsize=(8, 8))
            ax_peaks.imshow(st.session_state['registration_image'], cmap='gray')
            
            # Define colors for each quadrant
            colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'cyan'}
            
            peaks_dict = st.session_state['detected_peaks']
            if peaks_dict:
                for quad_idx, peaks in peaks_dict.items():
                    if peaks is not None and len(peaks) > 0:
                        color = colors.get(quad_idx, 'white')
                        ax_peaks.plot(peaks[:, 0], peaks[:, 1], '+', color=color, markersize=8, 
                                      label=f'Quad {quad_idx}: {len(peaks)} peaks')

            ax_peaks.set_title("All Initially Detected Fiducials")
            ax_peaks.axis('off')
            ax_peaks.legend()
            st.pyplot(fig_peaks)
            
            with st.expander("Show Processing Log"):
                st.text(st.session_state['log'])
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, st.session_state['output_stack'], imagej=True)
            buffer.seek(0)
            st.download_button(
                label="💾 Download Registered 4-Channel TIFF", data=buffer, file_name="registered_output.tif", mime="image/tiff", use_container_width=True
            )
        with res_col2:
            st.subheader("Registered Data Channels")
            stack = st.session_state['output_stack']
            fig_channels, axes_channels = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes_channels.flat):
                channel_data = stack[i]
                if channel_data.max() > 0:
                    norm_data = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    ax.imshow(norm_data, cmap='viridis')
                else:
                    ax.imshow(np.zeros_like(channel_data), cmap='gray', vmin=0, vmax=255)
                ax.set_title(f"Channel {i+1} (From Quad {i+1})")
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig_channels)
    elif 'log' in st.session_state:
        st.error("Registration failed. Please check the log for details.")
        with st.expander("Show Processing Log"):
            st.text(st.session_state['log'])
