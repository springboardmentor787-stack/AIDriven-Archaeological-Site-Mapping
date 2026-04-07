import streamlit as st
import cv2
import numpy as np

def render_deforestation_tab(img_rgb):
    st.subheader("🌲 Digital Deforestation")

    if img_rgb is None:
        st.warning("Upload an image in Analysis tab first.")
        return

    # Simple vegetation mask (demo logic)
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    vari = (g - r) / (g + r - b + 1e-6)

    veg_mask = vari > 0.3

    # Ground exposure
    ground = img_rgb.copy()
    ground[veg_mask] = [0, 0, 0]

    # Stats
    veg_pct = round(np.mean(veg_mask) * 100, 2)
    ground_pct = round(100 - veg_pct, 2)

    st.metric("Vegetation %", f"{veg_pct}%")
    st.metric("Ground Exposure %", f"{ground_pct}%")

    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="Original")
    col2.image(ground, caption="Deforestation View")

    st.success("Hidden structures can be analyzed from exposed ground.")