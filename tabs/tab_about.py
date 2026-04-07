import streamlit as st

def render_about_tab():
    st.subheader("ℹ️ About ArchAI")

    st.markdown("""
### 🛰️ ArchAI — Archaeological Intelligence Platform

ArchAI is an AI-powered system designed to analyze satellite imagery and detect potential archaeological sites.

---

### 🚀 Key Features
- Object Detection (YOLO-based)
- Vegetation Analysis (VARI Index)
- Erosion Risk Assessment
- Digital Deforestation
- Heatmap Visualization
- AI Report Generation

---

### 🎯 Purpose
To reduce:
- ⏳ Survey time
- 💰 Field costs
- ❌ Manual effort

---

### 🧠 Technology Stack
- Python
- OpenCV
- YOLO
- Streamlit
- GIS Mapping

---

### 👨‍💻 Developed By
**Hari Krishanan M**

---

### 🔥 Vision
Transform satellite data into actionable archaeological insights.
""")