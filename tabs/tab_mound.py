# ============================================================
# tabs/tab_mound.py — Object Detection / Survey Optimisation tab
# ============================================================
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from modules.mound_detection import (
    run_mound_pipeline, draw_mound_overlay,
    build_detection_heatmap, compute_cost_savings,
)
from widgets.mound_report import build_mound_report_widget


def render(params: dict, det_model, det_mode):
    st.markdown("## AI-Assisted Survey Optimization & Object Detection")
    st.caption(
        "Upload satellite or drone imagery — the AI detects ALL visible objects, classifies them "
        "as Man-made or Natural, filters out irrelevant zones, and estimates survey time/cost saved."
    )

    if det_model is None:
        st.info(
            "Demo mode — YOLO model not loaded. Image-driven synthetic candidates will be generated "
            "based on texture analysis to demonstrate the classification pipeline."
        )

    col_up, col_ctrl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload satellite / drone image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="mound_upload",
        )
    with col_ctrl:
        survey_area = st.number_input("Survey Area (sq.km)",
                                      min_value=1.0, max_value=500.0, value=50.0, step=5.0)
        filter_mode = st.checkbox("Show only Man-made candidates", value=True)
        run_btn     = st.button("Run Detection", use_container_width=True)

    if not uploaded:
        st.caption("Accepted formats: JPG, PNG, TIF.")
        st.markdown("""
        <div style="display:flex;gap:16px;margin-top:8px;font-family:'JetBrains Mono',monospace;
                    font-size:10px;letter-spacing:0.08em;">
          <span style="color:#d46b6b;">■ Man-made</span>
          <span style="color:#7ec899;">■ Natural</span>
          <span style="color:#d4a84b;">■ Uncertain</span>
        </div>""", unsafe_allow_html=True)
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

    if run_btn or st.session_state.get("mound_results"):
        if run_btn:
            with st.spinner("Running detection and classification…"):
                results = run_mound_pipeline(rgb, det_model, params["conf"], filter_high_conf=filter_mode)
            st.session_state["mound_results"] = results
        else:
            results = st.session_state.get("mound_results", [])

        if not results:
            st.warning("No candidates detected. Try lowering the confidence threshold.")
            return

        savings = compute_cost_savings(results, total_area_sqkm=survey_area)

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown("### Survey Cost & Time Savings")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Detected",     savings["total"])
        m2.metric("Man-made",           savings["manmade"],  delta="Priority")
        m3.metric("Natural (filtered)", savings["natural"],  delta=f"-{savings['pct_filtered']:.0f}% area")
        m4.metric("Uncertain",          savings["uncertain"])
        m5.metric("Days Saved",         savings["days_saved"], delta="vs traditional")
        m6.metric("Cost Saved (USD)",   f"${savings['cost_saved']:,}")

        st.markdown(f"""
        <div style="margin:8px 0 16px;padding:10px 14px;background:var(--bg-elevated);
                    border:1px solid var(--border-light);border-radius:4px;
                    font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);
                    letter-spacing:0.07em;">
          Traditional survey: <strong style="color:var(--accent);">${savings['cost_trad']:,}</strong> &nbsp;|&nbsp;
          AI-optimised: <strong style="color:var(--risk-low-fg);">${savings['cost_ai']:,}</strong> &nbsp;|&nbsp;
          Priority area: <strong style="color:var(--accent);">{savings['area_priority']} sq.km</strong>
          of {survey_area:.0f} sq.km &nbsp;|&nbsp;
          {savings['pct_filtered']:.1f}% filtered as natural
        </div>""", unsafe_allow_html=True)

        # ── Detection overlay ──────────────────────────────────────────────────
        st.markdown("### Detection Overlay")
        ov_img   = draw_mound_overlay(rgb, results, filter_high_conf=filter_mode)
        heat_img = build_detection_heatmap(rgb, results)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Original Image")
            st.image(rgb, use_container_width=True)
        with c2:
            st.markdown("#### Classification Overlay")
            st.image(ov_img, use_container_width=True)
            st.markdown("""<div style="display:flex;gap:12px;font-family:'JetBrains Mono',monospace;
              font-size:9px;letter-spacing:0.07em;margin-top:4px;">
              <span style="color:#d46b6b;">■ Man-made (thick border + label)</span>
              <span style="color:#7ec899;">■ Natural (thin border)</span>
              <span style="color:#d4a84b;">■ Uncertain (thin border)</span>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("#### Detection Density Heatmap")
            st.image(heat_img, use_container_width=True)
            st.caption("Brighter = higher detection density / confidence")

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("### Classification Results")
        df_rows = [{
            "#":         i + 1,
            "Label":     r["label"],
            "Class":     r["cls_label"],
            "Conf":      f"{r['conf']:.2%}",
            "Cls Score": f"{r['cls_score']:.3f}",
            "Shape Reg": f"{r['shape_reg']:.3f}",
            "Tex Var":   f"{r['tex_var']:.3f}",
            "VARI":      f"{r['vari_val']:.3f}",
            "Priority":  "✓" if r.get("highlight") else "—",
        } for i, r in enumerate(results)]
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Downloads ──────────────────────────────────────────────────────────
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            report_lines = [
                "SURVEY OPTIMIZATION REPORT", "=" * 44,
                f"Site         : {st.session_state.get('location_name', '—')}",
                f"Coordinates  : {params['lat']:.6f} N, {params['lon']:.6f} E",
                f"Survey Area  : {survey_area:.0f} sq.km", "",
                "DETECTION SUMMARY", "-" * 30,
                f"  Total Detected       : {savings['total']}",
                f"  Man-made Candidates  : {savings['manmade']}",
                f"  Natural              : {savings['natural']}",
                f"  Uncertain            : {savings['uncertain']}",
                f"  Area Filtered Out    : {savings['pct_filtered']}%",
                f"  Priority Area        : {savings['area_priority']} sq.km", "",
                "COST & TIME SAVINGS", "-" * 30,
                f"  Traditional (USD)    : ${savings['cost_trad']:,}",
                f"  AI-Optimised (USD)   : ${savings['cost_ai']:,}",
                f"  Estimated Saving     : ${savings['cost_saved']:,}",
                f"  Field Days Saved     : {savings['days_saved']}",
            ]
            st.download_button("Download Survey Report (.txt)",
                               data="\n".join(report_lines),
                               file_name="survey_report.txt", mime="text/plain")
        with col_dl2:
            st.download_button("Download Detection Data (.csv)", data=df.to_csv(index=False),
                               file_name="detections.csv", mime="text/csv")

        # ── AI report widget ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### AI Field Survey Report")
        st.components.v1.html(
            build_mound_report_widget(
                st.session_state.get("location_name", ""),
                params["lat"], params["lon"], savings, results,
            ),
            height=440, scrolling=False,
        )

        with st.expander("Classification Methodology"):
            st.markdown("""
| Feature | Weight | Description |
|---------|--------|-------------|
| Shape Regularity | 35% | Aspect ratio — man-made structures tend toward regular shapes |
| Texture Variance | 25% | Laplacian variance — natural objects have high texture |
| VARI Index | 25% | Vegetation — man-made surfaces typically lower vegetation |
| Detection Confidence | 15% | Model certainty score |

Score ≥ 0.65 → **Man-made** · Score ≤ 0.45 → **Natural** · Else → **Uncertain**
            """)
    else:
        st.caption("Upload an image and click **Run Detection** to begin.")