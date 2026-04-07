import streamlit as st
from config.settings import APP_TITLE, APP_ICON, APP_VERSION, APP_AUTHOR, SESSION_DEFAULTS
from config.styles import GLOBAL_CSS
from config import THEMES
from utils.model_loaders import load_local_yolo, load_erosion_model
from tabs.sidebar import render_sidebar
import tabs.tab_analysis as tab_analysis
import tabs.tab_mound as tab_mound
import tabs.tab_deforestation as tab_deforestation
import tabs.tab_map as tab_map
import tabs.tab_reports as tab_reports
import tabs.tab_about as tab_about

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON,
                   layout="wide", initial_sidebar_state="expanded")

for k, v in SESSION_DEFAULTS:
    if k not in st.session_state:
        st.session_state[k] = v

t = st.session_state.get("theme", "Dark")
vars_css = "\n".join(f"        {k}: {v};" for k, v in THEMES[t].items())
st.markdown(f"<style>:root {{{vars_css}}}</style>", unsafe_allow_html=True)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

params = render_sidebar()
det_model, det_mode  = load_local_yolo(params["local_path"]) if params["local_path"] else (None, None)
er_model, feat_names = load_erosion_model(params["erosion_path"])

st.markdown(f"""
<div class="title-bar">
  <div class="page-title">ArchAI</div>
  <div class="page-subtitle">Archaeological Intelligence Platform &middot; {APP_AUTHOR} &middot; v{APP_VERSION}</div>
</div>""", unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Analysis", "Object Detection", "Deforestation AI", "Map", "Reports", "About"]
)
with t1: tab_analysis.render(params, det_model, det_mode, er_model, feat_names)
with t2: tab_mound.render(params, det_model, det_mode)
with t3: tab_deforestation.render(params)
with t4: tab_map.render(params)
with t5: tab_reports.render(params)
with t6: tab_about.render()