import streamlit as st
import folium
from streamlit_folium import st_folium

def render_map_tab(lat, lon, detections):
    st.subheader("🗺️ Map Visualization")

    m = folium.Map(location=[lat, lon], zoom_start=10)

    # Main location
    folium.Marker(
        [lat, lon],
        popup="Survey Location",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # Detection markers
    for i, d in enumerate(detections):
        folium.Marker(
            [lat + i*0.001, lon + i*0.001],
            popup=f"{d['label']} ({d['conf']:.2f})",
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_folium(m, width=700, height=500)