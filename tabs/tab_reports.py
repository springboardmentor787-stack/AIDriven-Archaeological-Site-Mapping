import streamlit as st
import pandas as pd
import json

def render_reports_tab(detections, risk, location):
    st.subheader("📄 Report Generation")

    if not detections:
        st.warning("No detections available.")
        return

    df = pd.DataFrame(detections)

    st.dataframe(df)

    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV",
        csv,
        "report.csv",
        "text/csv"
    )

    # JSON Download
    report = {
        "location": location,
        "risk_score": risk,
        "detections": detections
    }

    st.download_button(
        "Download JSON",
        json.dumps(report, indent=2),
        "report.json"
    )

    st.success("Report ready for download.")