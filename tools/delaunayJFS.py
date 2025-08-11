import streamlit as st
import numpy as np
import cv2
import tifffile
import io
import contextlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# ==============================================================================
#  HELPER FUNCTIONS (UNCHANGED)
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
    quadrants = {
        1: image[0:h_half, 0:w_half], 2: image[0:h_half, w_half:w],
        3: image[h_half:h, 0:w_half], 4: image[h_half:h, w_half:w]
    }
    return quadrants.get(quad_index)

def getQuadrantOffset(image_shape, quad_index):
    # This function is unchanged.
    h, w = image_shape[:2]
    h_half, w_half = h // 2, w // 2
    offsets = {
        1: (0, 0), 2: (w_half, 0), 3: (0, h_half), 4: (w_half, h_half)
    }
    return offsets.get(quad_index, (0, 0))

# ==============================================================================
#  MAIN REGISTRATION WORKFLOW (UNCHANGED)
# ==============================================================================

def run_registration_workflow(imgRegistration, imgData, activeQuads, peakParams, matchParams):
    """Encapsulates the entire registration process."""
    # This function is unchanged.
    print("\nCalculating transformations from registration image...")
    refQuadIndex = activeQuads[0]
    sourceQuadIndices = [q for q in activeQuads if q != refQuadIndex]
    
    refQuadImg = getQuadrant(imgRegistration, refQuadIndex)
    refOffset = getQuadrantOffset(imgRegistration.shape, refQuadIndex)
    refPeaks, _ = findOptimalPeaks(refQuadImg, **peakParams)
    if len(refPeaks) == 0:
        print(f"Error: Could not find any peaks in the reference quadrant {refQuadIndex}. Aborting.")
        return None
    refPeaksGlobal = refPeaks + refOffset
    
    transformations = {}
    for srcQuadIndex in sourceQuadIndices:
        print(f"\nProcessing source quadrant {srcQuadIndex}...")
        srcQuadImg = getQuadrant(imgRegistration, srcQuadIndex)
        srcOffset = getQuadrantOffset(imgRegistration.shape, srcQuadIndex)
        srcPeaks, _ = findOptimalPeaks(srcQuadImg, **peakParams)
        if len(srcPeaks) == 0:
            print(f"  Could not find peaks in quadrant {srcQuadIndex}. Skipping.")
            continue
        srcPeaksGlobal = srcPeaks + srcOffset
        print(f"  Matching {len(srcPeaks)} source peaks to {len(refPeaks)} reference peaks...")
        matchedSourcePoints, matchedTargetPoints = matchFeatures(imgRegistration, srcPeaksGlobal, imgRegistration, refPeaksGlobal, **matchParams)
        print(f"  Found {len(matchedSourcePoints)} matching points.")
        if len(matchedSourcePoints) >= 3:
            transformations[srcQuadIndex] = (matchedSourcePoints, matchedTargetPoints)
        else:
            print(f"  Not enough matches to calculate transform for quadrant {srcQuadIndex}. Skipping.")

    print("\nApplying transformations to data image and building output file...")
    quad_h, quad_w = imgData.shape[0] // 2, imgData.shape[1] // 2
    output_stack = np.zeros((4, quad_h, quad_w), dtype=imgData.dtype)
    
    refDataQuad = getQuadrant(imgData, refQuadIndex)
    output_stack[refQuadIndex - 1] = refDataQuad
    print(f"Quadrant {refQuadIndex} (reference) placed in channel {refQuadIndex - 1}.")
    
    for srcQuadIndex, (srcPoints, tgtPoints) in transformations.items():
        print(f"Warping data for quadrant {srcQuadIndex}...")
        warpedFullDataImage = warpImagePiecewise(imgData, srcPoints, tgtPoints)
        ref_x, ref_y = getQuadrantOffset(imgData.shape, refQuadIndex)
        warpedData = warpedFullDataImage[ref_y:ref_y+quad_h, ref_x:ref_x+quad_w]
        output_stack[srcQuadIndex - 1] = warpedData
        print(f"Warped data from quadrant {srcQuadIndex} placed in channel {srcQuadIndex - 1}.")
    
    print("\nProcessing complete.")
    return output_stack
def run():
    # ==============================================================================
    #  STREAMLIT GUI
    # ==============================================================================
    
    st.set_page_config(layout="wide")
    st.title("🔬 Quadrant-Based Image Cross-Registration")
    st.markdown("---")
    
    # --- Define layout columns ---
    col1, col2 = st.columns(2)
    
    # --- Populate Column 1: Registration Setup ---
    with col1:
        st.subheader("1. Registration Setup")
        f_reg = st.file_uploader("Upload Registration Image (with fiducials)", type=["tif", "tiff"])
    
        with st.expander("Set Registration Parameters", expanded=True):
            activeQuads = st.multiselect(
                "Active quadrants (first is reference)",
                options=[1, 2, 3, 4],
                default=[1, 2, 3, 4]
            )
            st.markdown("###### Peak Detection")
            minPeaks = st.slider("Min desired peaks", 5, 500, 10, key="min_peaks")
            maxPeaks = st.slider("Max desired peaks", 10, 1000, 200, key="max_peaks")
            minRoundness = st.slider("Min peak roundness", 0.0, 1.0, 0.75, 0.05, key="roundness")
    
            st.markdown("###### Feature Matching")
            patchSize = st.slider("Match patch size", 5, 51, 21, step=2, key="patch_size")
            searchRadius = st.slider("Match search radius (pixels)", 10, 500, 100, key="search_radius")
    
    # --- Populate Column 2: Data Setup ---
    with col2:
        st.subheader("2. Data Setup")
        f_data = st.file_uploader("Upload Data Image (to be warped)", type=["tif", "tiff"])
    
    st.markdown("---")
    
    # --- Run Button and Results Display ---
    if f_reg and f_data:
        if st.button("🚀 Run Registration", use_container_width=True):
            imgRegistration = tifffile.imread(io.BytesIO(f_reg.read()))
            imgData = tifffile.imread(io.BytesIO(f_data.read()))
            peakParams = {'minPeaks': minPeaks, 'maxPeaks': maxPeaks, 'minRoundness': minRoundness}
            matchParams = {'patchSize': patchSize, 'searchRadius': searchRadius}
    
            log_stream = io.StringIO()
            with st.spinner("Registration in progress... this may take a moment."):
                with contextlib.redirect_stdout(log_stream):
                    output_stack = run_registration_workflow(
                        imgRegistration, imgData, activeQuads, peakParams, matchParams
                    )
            
            st.session_state['output_stack'] = output_stack
            st.session_state['log'] = log_stream.getvalue()
    
    # --- Display Results ---
    if 'output_stack' in st.session_state and st.session_state['output_stack'] is not None:
        st.success("✅ Registration complete!")
        
        # Display the processing log in a column to keep layout consistent
        log_col, res_col = st.columns([1, 2])
        with log_col:
            with st.expander("Show Processing Log"):
                st.text(st.session_state['log'])
    
            # Prepare data for download button
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, st.session_state['output_stack'], imagej=True)
            buffer.seek(0)
            
            st.download_button(
                label="💾 Download Registered 4-Channel TIFF",
                data=buffer,
                file_name="registered_output.tif",
                mime="image/tiff",
                use_container_width=True
            )
    
        with res_col:
            stack = st.session_state['output_stack']
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle('Registered Output Channels', fontsize=16)
            for i, ax in enumerate(axes.flat):
                channel_data = stack[i]
                if channel_data.max() > 0:
                    norm_data = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    ax.imshow(norm_data, cmap='viridis')
                else:
                    ax.imshow(np.zeros_like(channel_data), cmap='gray', vmin=0, vmax=255)
                ax.set_title(f"Channel {i+1} (From Quad {i+1})")
                ax.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
    
    elif 'log' in st.session_state:
        st.error("Registration failed. Please check the log for details.")
        with st.expander("Show Processing Log"):
            st.text(st.session_state['log'])
