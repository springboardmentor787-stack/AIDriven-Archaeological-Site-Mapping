# ============================================================
# tabs/sidebar.py — Sidebar rendering
# ============================================================
from pathlib import Path
import streamlit as st
from utils.geocoding import geocode_location
from widgets.ai_report import build_ai_report_widget


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: block !important; visibility: visible !important;
            opacity: 1 !important; transform: translateX(0) !important;
            min-width: 21rem !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: 0 !important; transform: translateX(0) !important;
            width: 21rem !important;
        }
        [data-testid="collapsedControl"] {
            display: flex !important; visibility: visible !important; opacity: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-brand">ArchAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Archaeological Intelligence Platform</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

        # ── Theme ──────────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Theme</div>', unsafe_allow_html=True)
        selected_theme = st.radio(
            "Theme", ["Dark", "Light"],
            index=0 if st.session_state["theme"] == "Dark" else 1,
            horizontal=True, label_visibility="collapsed",
        )
        if selected_theme != st.session_state["theme"]:
            st.session_state["theme"] = selected_theme
            st.rerun()

        # ── Detection model ────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Detection Model</div>', unsafe_allow_html=True)
        local_path = st.text_input("Weights path", value="model/best.pt",
                                   label_visibility="collapsed")
        st.caption("Model found" if Path(local_path.strip()).exists() else "Demo mode — model not found")
        conf = st.slider("Confidence threshold", 10, 90, 40, 5)

        # ── Erosion model ──────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Erosion Model</div>', unsafe_allow_html=True)
        erosion_path = st.text_input("Erosion model (.pkl)", value="erosion_model.pkl",
                                     label_visibility="collapsed")
        st.caption("Erosion model found"
                   if erosion_path and Path(erosion_path.strip()).exists()
                   else "Formula fallback active")

        # ── Location search ────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Location Search</div>', unsafe_allow_html=True)
        location_input = st.text_input(
            "Location name",
            value=st.session_state.get("location_name", ""),
            placeholder="e.g. Hampi, Karnataka",
            label_visibility="collapsed",
        )
        st.caption("Add district for small villages")

        if st.button("Locate Coordinates", use_container_width=True):
            if not location_input.strip():
                st.warning("Enter a location name.")
            else:
                with st.spinner("Searching…"):
                    lat, lon, msg = geocode_location(location_input)
                if lat is not None:
                    st.session_state.update({"lat": lat, "lon": lon,
                                             "location_name": location_input, "geo_msg": msg})
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # ── Coordinates ────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Coordinates</div>', unsafe_allow_html=True)
        lat = st.number_input("Latitude",  value=float(st.session_state.get("lat", 20.5937)),
                              format="%.6f")
        lon = st.number_input("Longitude", value=float(st.session_state.get("lon", 78.9629)),
                              format="%.6f")
        if lat != st.session_state.get("lat"): st.session_state["lat"] = lat
        if lon != st.session_state.get("lon"): st.session_state["lon"] = lon

        # ── Terrain parameters ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Terrain Parameters</div>', unsafe_allow_html=True)
        auto      = st.session_state.get("auto_terrain", {})
        slope     = auto.get("slope", 0.0)
        elev      = auto.get("elevation", 0.0)
        auto_conf = auto.get("confidence", 0.0)
        if auto and (slope > 0 or elev > 0):
            c1, c2, c3 = st.columns(3)
            c1.metric("Slope", f"{slope:.1f}°")
            c2.metric("Elev",  f"{elev:.0f}m")
            c3.metric("Conf",  f"{auto_conf:.0%}")
        else:
            st.caption("Upload an image in Analysis to auto-detect terrain.")

        # ── AI report widget ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">AI Analysis</div>', unsafe_allow_html=True)
        risk      = st.session_state.get("risk", 0.0)
        vari_mean = st.session_state.get("vari_mean", 0.0)
        coverage  = st.session_state.get("coverage", {})
        dets      = st.session_state.get("dets", [])
        loc_name  = st.session_state.get("location_name", "Unknown Site")
        risk_lbl  = "LOW" if risk < 0.33 else ("MODERATE" if risk < 0.66 else "HIGH")

        st.components.v1.html(
            build_ai_report_widget(loc_name, lat, lon, slope, elev, risk, risk_lbl,
                                   vari_mean, coverage, len(dets), auto_conf),
            height=510, scrolling=False,
        )

    return dict(local_path=local_path, conf=conf, erosion_path=erosion_path,
                lat=lat, lon=lon, slope=slope, elev=elev)