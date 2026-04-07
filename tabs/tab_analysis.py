# ============================================================
# tabs/tab_analysis.py — Image Analysis tab
# ============================================================
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from modules.image_processing import (
    run_detection, compute_vari, colorise_vari,
    segment_vegetation, predict_erosion_score, auto_detect_terrain,
)


def render(params: dict, det_model, det_mode, er_model, feat_names):
    st.markdown("## Image Analysis")

    if det_model is None:
        st.info("Demo mode active — YOLO model not loaded. VARI and erosion analysis are fully operational.")
    else:
        st.success(f"Detection model loaded — {params['local_path']}")

    uploaded = st.file_uploader("Upload satellite or drone image",
                                type=["jpg", "jpeg", "png", "tif", "tiff"])
    if not uploaded:
        st.caption("Accepted formats: JPG, PNG, TIF. Maximum recommended resolution: 4000 px.")
        return

    raw = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot decode image.")
        return

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("Detecting terrain parameters…"):
        terrain = auto_detect_terrain(rgb)
    st.session_state["auto_terrain"] = terrain

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Original Image")
        st.image(rgb, use_container_width=True)
    with c2:
        st.markdown("#### Artifact Detection")
        with st.spinner("Running inference…"):
            det_img, dets = run_detection(img, det_model, det_mode or "local", params["conf"])
        st.image(det_img, use_container_width=True)
        st.metric("Artifacts detected", len(dets))

    st.session_state["dets"] = dets

    if dets:
        st.dataframe(
            pd.DataFrame([{"Label": d["label"], "Confidence": f"{d['conf']:.2%}",
                           "Centre": f"({d['cx']}, {d['cy']})"} for d in dets]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    vari = compute_vari(rgb)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### VARI Vegetation Index")
        st.image(colorise_vari(vari), use_container_width=True)
        vari_mean = float(vari.mean())
        st.metric("Mean VARI", f"{vari_mean:.3f}")
        st.caption("Very low vegetation." if vari_mean < -0.05
                   else ("Sparse vegetation." if vari_mean < 0.2
                         else "Moderate to dense vegetation."))

    seg, cov = segment_vegetation(vari)
    with c4:
        st.markdown("#### Vegetation Segmentation")
        st.image(seg, use_container_width=True)
        for lbl, pct in cov.items():
            st.progress(int(pct), text=f"{lbl}  —  {pct}%")

    st.session_state["vari_mean"] = float(vari.mean())
    st.session_state["coverage"]  = cov

    st.markdown("---")
    st.markdown("## Erosion Risk Assessment")

    auto_slope = float(terrain["slope"])
    auto_elev  = float(terrain["elevation"])
    auto_conf  = float(terrain["confidence"])
    ndvi_val   = float(vari.mean())

    risk = predict_erosion_score(er_model, feat_names, auto_slope, auto_elev, ndvi_val)
    st.session_state["risk"] = risk

    lbl = "LOW" if risk < 0.33 else ("MODERATE" if risk < 0.66 else "HIGH")
    fg  = "#7ec899" if risk < 0.33 else ("#d4a84b" if risk < 0.66 else "#d46b6b")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Risk Score",           f"{risk:.3f}")
    m2.metric("Risk Level",           lbl)
    m3.metric("Slope",                f"{auto_slope:.1f}°")
    m4.metric("Elevation",            f"{auto_elev:.0f} m")
    m5.metric("Detection Confidence", f"{auto_conf:.0%}")

    st.markdown(f"""
    <div style="margin:12px 0 4px;height:6px;background:var(--border);border-radius:3px;overflow:hidden;">
      <div style="height:100%;width:{min(risk*100,100):.1f}%;background:{fg};border-radius:3px;"></div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text-muted);
                letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
      Risk score: {risk:.3f} / 1.000 &nbsp;|&nbsp; Classification: {lbl}
    </div>""", unsafe_allow_html=True)

    with st.expander("Terrain Detection Methodology"):
        st.markdown(
            f"| Parameter | Value | Detection Method |\n"
            f"|-----------|-------|------------------|\n"
            f"| Slope | {terrain['slope']}° | Sobel gradient magnitude |\n"
            f"| Elevation | {terrain['elevation']} m | Brightness + texture + shadow composite |\n"
            f"| VARI (NDVI proxy) | {ndvi_val:.3f} | R/G/B visible vegetation index |\n"
            f"| Detection confidence | {terrain['confidence']:.0%} | Image contrast and dynamic range |"
        )