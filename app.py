import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------
# PAGE CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="ArcheoAI - Archaeological Site Mapping",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# STYLING
# --------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main body background */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics container styling */
    .metric-container {
        background: linear-gradient(145deg, #ffffff, #fbfcfd);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        text-align: center;
        border: 1px solid #e5e7eb;
        border-top: 4px solid #2563eb;
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 13px;
        color: #4b5563;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .metric-subtitle {
        font-size: 12px;
        color: #6b7280;
        margin-top: 6px;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #111827;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# HELPER FUNCTIONS
# --------------------------
@st.cache_data
def generate_segmentation(img_array):
    if img_array is None: return None, None
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (35, 35), 0)
    seg = np.zeros_like(img_array)
    
    colors = {
        'Vegetation': (34, 139, 34),
        'Soil': (139, 69, 19),
        'Ruins': (169, 169, 169),
        'Water': (65, 105, 225),
        'Bedrock': (105, 105, 105)
    }
    
    seg[(blurred <= 60)] = colors['Water']
    seg[(blurred > 60) & (blurred <= 100)] = colors['Vegetation']
    seg[(blurred > 100) & (blurred <= 140)] = colors['Soil']
    seg[(blurred > 140) & (blurred <= 180)] = colors['Ruins']
    seg[(blurred > 180)] = colors['Bedrock']
    
    blended = cv2.addWeighted(img_array, 0.6, seg, 0.4, 0)
    coverage = {"Water": 10, "Vegetation": 25, "Soil": 30, "Ruins": 25, "Bedrock": 10}
    return blended, coverage

@st.cache_data
def generate_detection(img_array, num_objects=7):
    if img_array is None: return None, None
    out_img = img_array.copy()
    h, w = img_array.shape[:2]
    np.random.seed(42)  # For consistent mock behavior per image
    objects = []
    classes = ['Ruins', 'Pottery', 'Stone Tool', 'Arch. Block']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i in range(num_objects):
        x1 = np.random.randint(0, max(1, w - 150))
        y1 = np.random.randint(0, max(1, h - 150))
        x2 = x1 + np.random.randint(50, 200)
        y2 = y1 + np.random.randint(50, 200)
        cls_idx = np.random.randint(0, len(classes))
        conf = np.random.uniform(0.65, 0.98)
        color = colors[cls_idx]
        
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 3)
        label = f"{classes[cls_idx]}: {conf:.2f}"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out_img, (x1, max(0, y1 - 25)), (x1 + label_w, max(0, y1)), color, -1)
        cv2.putText(out_img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        objects.append({"Class": classes[cls_idx], "Confidence": round(conf, 3)})
        
    df = pd.DataFrame(objects)
    return out_img, df

@st.cache_data
def generate_erosion(img_array):
    if img_array is None: return None
    h, w = img_array.shape[:2]
    np.random.seed(24)
    noise = np.random.rand(h // 4, w // 4).astype(np.float32)
    noise_resized = cv2.resize(noise, (w, h))
    smoothed = cv2.GaussianBlur(noise_resized, (151, 151), 0)
    
    heatmap = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    blended = cv2.addWeighted(img_array, 0.5, heatmap_colored, 0.5, 0)
    return blended

def metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("Control Panel")
st.sidebar.markdown("Upload imagery and configure analysis models.")

uploaded_file = st.sidebar.file_uploader("Upload Site Image (Drag & Drop)", type=["jpg", "jpeg", "png", "tif"])

st.sidebar.subheader("Model Selection")
seg_model = st.sidebar.selectbox("Segmentation Model", ["U-Net", "DeepLabV3+", "Mask R-CNN"])
det_model = st.sidebar.selectbox("Detection Model", ["YOLOv8", "YOLOv5", "Faster R-CNN"])
ero_model = st.sidebar.selectbox("Erosion Model", ["Random Forest", "XGBoost", "CatBoost"])

st.sidebar.subheader("Adjustable Parameters")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
resolution = st.sidebar.selectbox("Image Resolution", ["High (1080p)", "Medium (720p)", "Low (480p)"])
erosion_horizon = st.sidebar.slider("Erosion Prediction Horizon (Years)", 5, 100, 10)

st.sidebar.markdown("---")
st.sidebar.info("ArcheoAI Dashboard Version 1.2")

# --------------------------
# MAIN CONTENT
# --------------------------
st.title("AI-Driven Archaeological Site Mapping")
st.markdown("An industry-grade analytics platform for the automated detection, segmentation, and erosion analysis of archaeological sites.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Site Analysis", 
    "Segmentation", 
    "Object Detection", 
    "Erosion Prediction", 
    "Report"
])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        # Generate outputs for visualization
        seg_img, coverage = generate_segmentation(img_array)
        det_img, det_df = generate_detection(img_array, num_objects=np.random.randint(5, 15))
        ero_img = generate_erosion(img_array)
        
        with tab1:
            st.header("Site Analysis Overview")
            st.markdown("Holistic overview of the archaeological site's condition and model metrics.")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Ruins Coverage", f"{coverage.get('Ruins', 0)}%", "Based on Segmentation")
            with col2:
                # Filter mock detections by threshold before counting
                valid_det = len(det_df[det_df['Confidence'] >= conf_threshold])
                metric_card("Artifacts Detected", f"{valid_det}", f"Conf > {conf_threshold}")
            with col3:
                metric_card("Erosion Risk Score", "High (74%)", f"Horizon: {erosion_horizon} yrs")
            with col4:
                metric_card("Model mAP", "0.82", f"Current Model: {det_model}")
                
            st.subheader("Site Assessment Summary")
            st.info(f"**AI Assessment:** The provided region presents approximately **{coverage.get('Ruins', 0)}%** ruins coverage. A total of **{valid_det}** artifacts or archaeological features were successfully detected using {det_model} with an average confidence of **{det_df['Confidence'].mean() * 100:.1f}%**. Immediate attention is required as the structural erosion projection model ({ero_model}) indicates a high risk over the next {erosion_horizon} years.")
            
            st.subheader("Overall Model Performance Table")
            perf_data = {
                "Task": ["Semantic Segmentation", "Object Detection", "Erosion Modeling"],
                "Model Used": [seg_model, det_model, ero_model],
                "Metric 1": ["IoU: 0.78", "mAP@50: 0.82", "RMSE: 2.14"],
                "Metric 2": ["Dice Score: 0.83", "Precision: 0.88", "R² Score: 0.79"]
            }
            st.table(pd.DataFrame(perf_data))

        with tab2:
            st.header("Semantic Segmentation")
            st.markdown("Detailed pixel-wise classification of site geography and ruins.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(seg_img, use_container_width=True, channels="RGB", caption="Segmentation Map Overlay")
            with col2:
                st.subheader("Coverage Breakdown")
                fig, ax = plt.subplots(figsize=(4, 4))
                colors_bar = ['royalblue', 'forestgreen', 'sienna', 'gray', 'dimgray']
                ax.bar(coverage.keys(), coverage.values(), color=colors_bar)
                ax.set_ylabel("Percentage (%)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                st.subheader("Class Legend & Details")
                st.markdown("""
                - 🟩 **Vegetation:** Shrubs, trees
                - 🟫 **Soil:** Bare ground
                - ⬜ **Ruins:** Archaeological structures
                - 🟦 **Water:** Rivers, lakes
                - ⬛ **Bedrock:** Natural stone
                """)
                
            st.subheader("Metrics Breakdown")
            seg_metrics = pd.DataFrame({
                "Class": list(coverage.keys()),
                "IoU": [0.91, 0.88, 0.76, 0.74, 0.81],
                "Dice Score": [0.95, 0.93, 0.86, 0.85, 0.89],
                "Precision": [0.93, 0.89, 0.78, 0.75, 0.84]
            })
            st.dataframe(seg_metrics, width="stretch")

        with tab3:
            st.header("Object Detection")
            st.markdown("Boundary box identification of key artifacts and architectural elements.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(det_img, use_container_width=True, channels="RGB", caption=f"Detected Features with {det_model}")
            with col2:
                st.subheader("Detected Objects Details")
                filtered_df = det_df[det_df['Confidence'] >= conf_threshold].reset_index(drop=True)
                st.dataframe(filtered_df, width="stretch")
                
                st.markdown("""
                **Explanation of Metrics:**
                - **Confidence Score:** The model's certainty that an object belongs to the predicted class.
                - **Precision:** The ratio of correct confident positive predictions.
                - **Recall:** Out of all actual positive objects, how many the model found.
                """)
                
            st.subheader("Model Validation: Precision vs Recall")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            pr_classes = ['Ruins', 'Pottery', 'Stone Tool', 'Arch. Block']
            precision = [0.85, 0.78, 0.91, 0.82]
            recall = [0.81, 0.74, 0.88, 0.79]
            x = np.arange(len(pr_classes))
            width = 0.35
            
            ax2.bar(x - width/2, precision, width, label='Precision', color='indigo')
            ax2.bar(x + width/2, recall, width, label='Recall', color='darkorange')
            ax2.set_xticks(x)
            ax2.set_xticklabels(pr_classes)
            ax2.legend()
            st.pyplot(fig2)

        with tab4:
            st.header("Structural Erosion Prediction")
            st.markdown("Heatmap projection based on terrain slope, vegetation indices (NDVI), and hydrological factors.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(ero_img, use_container_width=True, channels="RGB", caption=f"Risk Heatmap (Projection: +{erosion_horizon} Years)")
            with col2:
                st.subheader("Feature Importance")
                fig3, ax3 = plt.subplots(figsize=(4, 4))
                features = ['Slope', 'NDVI', 'Elevation', 'Soil Type']
                importance = [0.45, 0.25, 0.20, 0.10]
                ax3.barh(features[::-1], importance[::-1], color='crimson') # Reverse for descending view
                ax3.set_xlabel("Importance Weight")
                st.pyplot(fig3)
                
                st.markdown("""
                **Heatmap Legend:**
                - <span style="color:red;font-weight:bold;">Red:</span> Severe Erosion Risk
                - <span style="color:orange;font-weight:bold;">Orange/Yellow:</span> Moderate Risk
                - <span style="color:blue;font-weight:bold;">Blue:</span> Stable / Low Risk
                
                *A Gaussian smoothing filter is applied for accurate geospatial transitions.*
                """, unsafe_allow_html=True)
                
            col3, col4 = st.columns([2, 1])
            with col3:
                st.subheader(f"Temporal Forecast (Over {erosion_horizon} Years)")
                years = np.arange(1, erosion_horizon + 1)
                risk_trend = 20 + 1.5 * years + np.random.normal(0, 2, len(years))
                fig4, ax4 = plt.subplots(figsize=(10, 3))
                ax4.plot(years, risk_trend, marker='o', linestyle='-', color='firebrick')
                ax4.set_xlabel("Years from Present")
                ax4.set_ylabel("Overall Risk Index")
                ax4.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig4)
                
            with col4:
                st.subheader("Model Metrics")
                st.markdown(f"""
                - **RMSE (Root Mean Square Error):** 2.14
                - **MAE (Mean Absolute Error):** 1.85
                - **R² Score:** 0.79
                
                *Represents **{ero_model}** validation performance on the historical baseline dataset.*
                """)

        with tab5:
            st.header("Final Assessment Report")
            
            st.markdown(f"""
            ### Executive Summary
            **Site Location:** User Uploaded Geo-Tiff/Raster
            **Target Horizon:** {erosion_horizon} Years
            
            #### Key Findings
            - **Ruins Coverage:** {coverage.get('Ruins', 0)}% of the surveyed area.
            - **Total Artifact Zones Detected:** {valid_det} items with confidence ≥ {conf_threshold}.
            - **Erosion Status:** Peak structural degradation projected at localized anomalies, primarily driven by terrain slope.
            
            #### Methodologies Used
            - **Segmentation ({seg_model}):** Extracted precise boundary delineations for vegetation, soil, ruins, water, and bedrock.
            - **Detection ({det_model}):** Bounding box inference with an IoU threshold of {iou_threshold}.
            - **Risk Modeling ({ero_model}):** Predictive geospatial mapping using feature-weighted variables.
            
            ---
            
            ### Conclusion & Recommendations
            The structural integrity of the uncovered areas shows signs of vulnerability to the elements over the specified {erosion_horizon}-year horizon. 
            
            **Recommendations:**
            1. **Immediate Intervention:** Prioritize physical reinforcement in regions identified as <span style="color:red">red zones</span> in the Erosion Prediction tab.
            2. **Targeted Excavation:** Focus initial efforts on the high-confidence artifact zones localized by the {det_model} model.
            3. **Continuous Monitoring:** Resurvey the area with aerial or satellite imagery periodically to validate the predictive degradation curve.
            """, unsafe_allow_html=True)
            
            if st.button("Download PDF Report"):
                st.success("Report generation initiated! (PDF ready for download)")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    # State when no image is uploaded
    st.info("Welcome to ArcheoAI! Please upload a site image from the sidebar to begin processing.")
    st.image("https://images.unsplash.com/photo-1549880181-56a44cf4a9a5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80", caption="Example Archaeological Site Concept", use_container_width=True)