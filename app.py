import streamlit as st
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import joblib
import requests
import certifi
import importlib
import matplotlib.pyplot as plt
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from terrain_model.real_feature_extractor import extract_real_features
from terrain_model.ai_explainer import generate_gemini_explanation, generate_gemini_from_prompt


def resolve_segmentation_checkpoint():
    candidates = [
        os.path.join("models", "deeplab_model.pth"),
        os.path.join("runs", "segmentation", "deeplab_model_best.pth"),
        "deeplab_model.pth",
    ]

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("No segmentation checkpoint found.")

    return max(existing, key=os.path.getmtime)


def resolve_erosion_model_path():
    candidates = [
        os.path.join("models", "erosion_model.pkl"),
        "erosion_model.pkl",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No erosion model checkpoint found.")

def generate_report(
    vegetation_ratio,
    slope,
    elevation,
    erosion_label,
    lat,
    lon,
    location_source,
    rainfall=None,
    soil_type=None,
    erosion_prob=None,
    risk_zone=None
):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Archaeological Site Analysis Report", styles['Title']))
    content.append(Paragraph(f"Vegetation Ratio: {vegetation_ratio:.2f}", styles['Normal']))
    content.append(Paragraph(f"Slope: {slope:.2f}", styles['Normal']))
    content.append(Paragraph(f"Elevation: {elevation:.2f}", styles['Normal']))
    if rainfall is not None:
        content.append(Paragraph(f"Rainfall: {rainfall:.2f} mm", styles['Normal']))
    if soil_type is not None:
        content.append(Paragraph(f"Soil Type: {soil_type.capitalize()}", styles['Normal']))
    content.append(Paragraph(f"Erosion Risk: {erosion_label}", styles['Normal']))
    if erosion_prob is not None:
        content.append(Paragraph(f"Erosion Probability: {erosion_prob * 100:.2f}%", styles['Normal']))
    if risk_zone is not None:
        content.append(Paragraph(f"Risk Zone: {risk_zone}", styles['Normal']))
    content.append(Paragraph(f"Latitude: {lat:.4f}", styles['Normal']))
    content.append(Paragraph(f"Longitude: {lon:.4f}", styles['Normal']))
    content.append(Paragraph(f"Location Source: {location_source}", styles['Normal']))

    doc.build(content)

def get_real_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=5, verify=certifi.where())
        data = response.json()
        return data["results"][0]["elevation"]
    except:
        print("Using fallback elevation")
        return np.random.uniform(350, 450)

def get_rainfall(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum"
        response = requests.get(url, timeout=5, verify=certifi.where())
        data = response.json()
        return data["daily"]["precipitation_sum"][0]
    except:
        print("Using fallback rainfall")
        return np.random.uniform(0, 50)

def get_avg_rainfall(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&past_days=7"
        response = requests.get(url, timeout=5, verify=certifi.where())
        data = response.json()
        rain = data["daily"]["precipitation_sum"]
        return sum(rain) / len(rain) if rain else 0.0
    except:
        print("Using fallback average rainfall")
        return np.random.uniform(0, 50)

def get_soil_type(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&depth=0-5cm"
        response = requests.get(url, timeout=5, verify=certifi.where())
        data = response.json()
        clay = data["properties"]["layers"][0]["depths"][0]["values"]["mean"]
        
        if clay > 40:
            return 3  # clay
        elif clay > 20:
            return 2  # loam
        else:
            return 1  # sandy
    except:
        print("Using fallback soil type")
        return np.random.choice([1, 2, 3])

def get_slope(lat, lon):
    try:
        e1 = get_real_elevation(lat, lon)
        e2 = get_real_elevation(lat + 0.001, lon)
        e3 = get_real_elevation(lat, lon + 0.001)

        dx = abs(e2 - e1)
        dy = abs(e3 - e1)

        slope = (dx + dy) / 2

        return slope
    except:
        print("Using fallback slope")
        return np.random.uniform(5, 30)

def get_image_gps(image):
    try:
        exif = image._getexif()
        if not exif:
            return None, None

        gps_info = {}

        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]

        def _to_float(part):
            try:
                return float(part)
            except (TypeError, ValueError):
                return float(part[0]) / float(part[1])

        def convert_to_degrees(value):
            d = _to_float(value[0])
            m = _to_float(value[1])
            s = _to_float(value[2])
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] != "N":
            lat = -lat

        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] != "E":
            lon = -lon

        return lat, lon

    except:
        return None, None

def load_map_modules():
    try:
        folium_lib = importlib.import_module("folium")
        st_folium_fn = importlib.import_module("streamlit_folium").st_folium
        plugins = importlib.import_module("folium.plugins")
        geocoder_cls = plugins.Geocoder
        return folium_lib, st_folium_fn, geocoder_cls
    except Exception:
        return None, None, None

def extract_map_coords(map_data):
    if not map_data:
        return None, None, None

    last_clicked = map_data.get("last_clicked")
    if isinstance(last_clicked, dict) and "lat" in last_clicked and "lng" in last_clicked:
        return float(last_clicked["lat"]), float(last_clicked["lng"]), "Map Click"

    last_object_clicked = map_data.get("last_object_clicked")
    if isinstance(last_object_clicked, dict) and "lat" in last_object_clicked and "lng" in last_object_clicked:
        return float(last_object_clicked["lat"]), float(last_object_clicked["lng"]), "Map Search"

    last_active = map_data.get("last_active_drawing")
    if isinstance(last_active, dict):
        geometry = last_active.get("geometry", {})
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            return float(coordinates[1]), float(coordinates[0]), "Map Search"

    all_drawings = map_data.get("all_drawings")
    if isinstance(all_drawings, list) and all_drawings:
        latest = all_drawings[-1]
        if isinstance(latest, dict):
            geometry = latest.get("geometry", {})
            coordinates = geometry.get("coordinates")
            if isinstance(coordinates, list) and len(coordinates) >= 2:
                return float(coordinates[1]), float(coordinates[0]), "Map Search"

    return None, None, None

def build_erosion_features(
    erosion_model,
    slope,
    vegetation_ratio,
    elevation,
    boulders_ratio,
    ruins_ratio,
    structures_ratio,
    rainfall=None,
    soil_value=None
):
    expected_features = getattr(erosion_model, "n_features_in_", None)

    if expected_features == 3:
        # Legacy model format: slope, vegetation_ratio, elevation
        return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3

    if expected_features == 4:
        # Intermediate model format: slope, vegetation_ratio, ruins_ratio, elevation
        return np.array([[slope, vegetation_ratio, ruins_ratio, elevation]], dtype=np.float32), 4

    if expected_features == 6:
        # Current model format: slope, vegetation_ratio, elevation, boulders_ratio, ruins_ratio, structures_ratio
        return np.array(
            [[slope, vegetation_ratio, elevation, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32
        ), 6

    if expected_features == 5:
        # Retrained model format: slope, vegetation, elevation, rainfall, soil
        rainfall = rainfall if rainfall is not None else 0.0
        soil_value = soil_value if soil_value is not None else 2
        return np.array(
            [[slope, vegetation_ratio, elevation, rainfall, soil_value]],
            dtype=np.float32
        ), 5

    if expected_features == 8:
        # Extended model format: slope, vegetation, elevation, rainfall, soil, boulders, ruins, structures
        rainfall = rainfall if rainfall is not None else 0.0
        soil_value = soil_value if soil_value is not None else 2
        return np.array(
            [[slope, vegetation_ratio, elevation, rainfall, soil_value, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32
        ), 8

    # Safe default for unknown models.
    return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3
# ------------------------
# Page Config
# ------------------------

st.set_page_config(
    page_title="Geo AI System",
    layout="wide"
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Sora:wght@500;700&display=swap');

:root {
    --bg-1: #06141f;
    --bg-2: #0a2230;
    --bg-3: #143247;
    --glass: rgba(255, 255, 255, 0.08);
    --glass-strong: rgba(255, 255, 255, 0.14);
    --line: rgba(255, 255, 255, 0.18);
    --text: #e9f4ff;
    --muted: #9ec0d6;
    --accent-a: #12c2a9;
    --accent-b: #1da1f2;
    --accent-c: #ff9f43;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 12%, rgba(18, 194, 169, 0.24), transparent 26%),
        radial-gradient(circle at 88% 18%, rgba(29, 161, 242, 0.22), transparent 28%),
        radial-gradient(circle at 50% 92%, rgba(255, 159, 67, 0.16), transparent 30%),
        linear-gradient(145deg, var(--bg-1), var(--bg-2) 52%, var(--bg-3));
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6, 20, 31, 0.96), rgba(8, 28, 40, 0.94));
    border-right: 1px solid var(--line);
}

section.main > div.block-container {
    padding-top: 1.4rem;
    max-width: 1280px;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.01em;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--glass);
    border: 1px solid var(--line);
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

[data-testid="stMetric"] [data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.06));
    border: 1px solid var(--line);
    border-radius: 14px;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    padding: 12px;
}

[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
}

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b), var(--accent-c));
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: linear-gradient(120deg, rgba(18, 194, 169, 0.22), rgba(29, 161, 242, 0.24));
    color: var(--text);
    font-weight: 600;
    transition: transform 180ms ease, box-shadow 180ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(29, 161, 242, 0.35);
}

.hero-shell {
    position: relative;
    padding: 24px 26px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.06));
    backdrop-filter: blur(10px);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
    overflow: hidden;
    animation: fadeUp 520ms ease-out;
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: -90px -80px auto auto;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(18, 194, 169, 0.45), transparent 66%);
}

.hero-title {
    margin: 0;
    font-size: 2.15rem;
    line-height: 1.2;
}

.hero-sub {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 1rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>

<div class="hero-shell">
  <h1 class="hero-title">🛰️ Archaeological Site Mapping AI</h1>
  <p class="hero-sub">AI-powered terrain and erosion analysis system</p>
</div>
""",
    unsafe_allow_html=True,
)

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

# ------------------------
# Load Models (Cached)
# ------------------------

@st.cache_resource
def load_models():

    yolo_model = YOLO("runs/detect/yolov8s_archaeology2/weights/best.pt")

    seg_model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6
    )

    seg_model.load_state_dict(
        torch.load(resolve_segmentation_checkpoint(), map_location="cpu")
    )

    seg_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = seg_model.to(device)

    erosion_model = joblib.load(resolve_erosion_model_path())

    return yolo_model, seg_model, erosion_model, device


yolo_model, seg_model, erosion_model, device = load_models()

# ------------------------
# Sidebar Controls
# ------------------------

st.sidebar.title("⚙️ Controls")

gemini_api_key = GEMINI_API_KEY

st.sidebar.subheader("📍 Location Selection")
if "manual_lat" not in st.session_state:
    st.session_state.manual_lat = 15.3350
if "manual_lon" not in st.session_state:
    st.session_state.manual_lon = 76.4600

# Apply pending map/geocoder selection before widgets are instantiated.
if "pending_map_lat" in st.session_state and "pending_map_lon" in st.session_state:
    st.session_state.manual_lat = float(st.session_state.pending_map_lat)
    st.session_state.manual_lon = float(st.session_state.pending_map_lon)
    del st.session_state["pending_map_lat"]
    del st.session_state["pending_map_lon"]

manual_lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    format="%.6f",
    key="manual_lat"
)
manual_lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    format="%.6f",
    key="manual_lon"
)

st.sidebar.subheader("🗺️ Select Location on Map")
folium_lib, st_folium_fn, geocoder_cls = load_map_modules()

if folium_lib and st_folium_fn and geocoder_cls:
    folium_map = folium_lib.Map(
        location=[manual_lat, manual_lon],
        zoom_start=6,
        tiles=None
    )
    folium_lib.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(folium_map)
    folium_lib.Marker([manual_lat, manual_lon], tooltip="Current Location").add_to(folium_map)
    folium_map.add_child(folium_lib.LatLngPopup())
    geocoder_cls(collapsed=False, add_marker=True).add_to(folium_map)

    folium_lib.LayerControl().add_to(folium_map)

    with st.sidebar:
        map_data = st_folium_fn(folium_map, height=300, width=300, key="location_map")

    selected_lat, selected_lon, selected_source = extract_map_coords(map_data)
    if selected_lat is not None and selected_lon is not None:
        if (
            abs(selected_lat - st.session_state.manual_lat) > 1e-9
            or abs(selected_lon - st.session_state.manual_lon) > 1e-9
        ):
            st.session_state.pending_map_lat = selected_lat
            st.session_state.pending_map_lon = selected_lon
            st.rerun()
        st.sidebar.success(f"📍 {selected_source}: {selected_lat:.4f}, {selected_lon:.4f}")
else:
    st.sidebar.info("Install `folium` and `streamlit-folium` to enable map picking.")

st.sidebar.subheader("Model Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    0.1, 0.9, 0.25
)

# Class toggles
show_vegetation = st.sidebar.checkbox("Vegetation", True)
show_ruins = st.sidebar.checkbox("Ruins", True)
show_structures = st.sidebar.checkbox("Structures", True)
show_boulders = st.sidebar.checkbox("Boulders", True)
show_others = st.sidebar.checkbox("Others", True)

CLASS_VISIBILITY = {
    "vegetation": show_vegetation,
    "ruins": show_ruins,
    "structures": show_structures,
    "boulders": show_boulders,
    "others": show_others
}

# ------------------------
# Colors
# ------------------------

CLASS_COLORS = {
    "boulders": (0,255,255),
    "others": (200,200,200),
    "ruins": (255,0,0),
    "structures": (0,0,255),
    "vegetation": (0,255,0)
}

SEG_COLORS = {
    0:[0,0,0],        # background
    1:[0,255,255],    # boulders
    2:[200,200,200],  # others
    3:[255,0,0],      # ruins
    4:[0,0,255],      # structures
    5:[0,255,0]       # vegetation
}

CLASS_ID_TO_NAME = {
    1:"boulders",
    2:"others",
    3:"ruins",
    4:"structures",
    5:"vegetation"
}

# ------------------------
# Upload Image
# ------------------------

st.markdown("### 📤 Input")
input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["jpg", "jpeg", "png"]
    )

with input_col2:
    st.caption("Using sidebar coordinates")

if uploaded_file:

    pil_image = Image.open(uploaded_file)
    image = np.array(pil_image.convert("RGB"))

    lat, lon = get_image_gps(pil_image)

    if lat is None or lon is None:
        st.info("ℹ️ No GPS data detected - using manual coordinates")
        lat, lon = manual_lat, manual_lon
        location_source = "Manual Input"
    else:
        st.success(f"📍 GPS detected: {lat:.4f}, {lon:.4f}")
        location_source = "Image EXIF GPS"

    # ------------------------
    # YOLO Detection
    # ------------------------

    detection_img = image.copy()

    results = yolo_model.predict(
        image,
        conf=confidence_threshold
    )

    for box in results[0].boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = yolo_model.names[cls]

        if not CLASS_VISIBILITY.get(label, True):
            continue

        color = CLASS_COLORS.get(label,(255,255,255))

        cv2.rectangle(detection_img,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            detection_img,
            f"{label} {conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # ------------------------
    # Segmentation
    # ------------------------

    img = cv2.resize(image,(512,512))

    img_tensor = torch.tensor(
        img.transpose(2,0,1)/255.0,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = seg_model(img_tensor)
        mask = torch.argmax(pred,dim=1).squeeze().cpu().numpy()

    mask = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1],image.shape[0])
    )

    seg_vis = image.copy()

    for class_id,color in SEG_COLORS.items():

        if class_id == 0:
            continue

        class_name = CLASS_ID_TO_NAME[class_id]

        if not CLASS_VISIBILITY.get(class_name,True):
            continue

        seg_vis[mask==class_id] = color

    seg_overlay = cv2.addWeighted(image,0.85,seg_vis,0.15,0)
    
    # ------------------------
    # Erosion Heatmap
    # ------------------------

    heatmap = np.zeros_like(image)

    # High erosion → RED
    heatmap[mask == 5] = [0, 0, 255]   # vegetation areas (low vegetation = risky)
    heatmap[mask == 3] = [255, 0, 0]   # ruins

    heat_overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    # ------------------------
    # Erosion Prediction
    # ------------------------

    boulders_pixels = np.sum(mask == 1)
    vegetation_pixels = np.sum(mask == 5)
    ruins_pixels = np.sum(mask == 3)
    structures_pixels = np.sum(mask == 4)
    total_pixels = mask.size

    boulders_ratio = boulders_pixels / total_pixels
    vegetation_ratio = vegetation_pixels / total_pixels
    ruins_ratio = ruins_pixels / total_pixels
    structures_ratio = structures_pixels / total_pixels

    # Extract real features using the terrain feature extractor
    features_dict = extract_real_features(lat, lon, vegetation_ratio)
    slope = features_dict["slope"]
    rainfall = features_dict["rainfall"]
    soil = features_dict["soil"]
    elevation = features_dict["elevation"]
    
    # Map soil value to soil type name for display
    soil_map_reverse = {1: "sandy", 2: "loam", 3: "clay"}
    soil_type = soil_map_reverse.get(soil, "loam")

    features, feature_mode = build_erosion_features(
        erosion_model,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil
    )

    if feature_mode == 3:
        st.caption("Model mode: 3 features (legacy)")
    elif feature_mode == 4:
        st.caption("Model mode: 4 features")
    elif feature_mode == 5:
        st.caption("Model mode: 5 features")
    elif feature_mode == 6:
        st.caption("Model mode: 6 features")
    elif feature_mode == 8:
        st.caption("Model mode: 8 features")

    prob = erosion_model.predict_proba(features)[0]
    erosion_prob = prob[1]

    erosion_label = "HIGH" if erosion_prob > 0.5 else "LOW"

    if erosion_prob < 0.3:
        risk_zone = "LOW RISK"
        risk_color = "#4CAF50"
    elif erosion_prob < 0.7:
        risk_zone = "MODERATE"
        risk_color = "#FFC107"
    else:
        risk_zone = "HIGH RISK"
        risk_color = "#F44336"

    # ------------------------
    # Combined Output
    # ------------------------

    combined = seg_overlay.copy()

    for box in results[0].boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = yolo_model.names[cls]

        if not CLASS_VISIBILITY.get(label,True):
            continue

        color = CLASS_COLORS.get(label,(255,255,255))

        cv2.rectangle(combined,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            combined,
            f"{label} {conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    ai_features = {
        "slope": slope,
        "vegetation": vegetation_ratio,
        "rainfall": rainfall,
        "soil": soil_type,
        "boulders": boulders_ratio,
        "ruins": ruins_ratio,
        "structures": structures_ratio,
    }
    ai_explanation = generate_gemini_explanation(ai_features, erosion_prob, api_key=gemini_api_key or None)

    top_features = []
    top_values = []
    shap_fig = None
    shap_error = None

    try:
        explainer = shap.Explainer(erosion_model)
        shap_values = explainer(features)

        feature_names = list(getattr(erosion_model, "feature_names_in_", []))
        if len(feature_names) != features.shape[1]:
            feature_names_by_mode = {
                3: ["slope", "vegetation", "elevation"],
                4: ["slope", "vegetation", "ruins", "elevation"],
                5: ["slope", "vegetation", "elevation", "rainfall", "soil"],
                6: ["slope", "vegetation", "elevation", "boulders", "ruins", "structures"],
                8: ["slope", "vegetation", "elevation", "rainfall", "soil", "boulders", "ruins", "structures"],
            }
            feature_names = feature_names_by_mode.get(
                feature_mode,
                ["slope", "vegetation", "elevation", "rainfall", "soil", "boulders", "ruins", "structures"][:features.shape[1]],
            )

        feature_name_aliases = {
            "vegetation_ratio": "vegetation",
            "boulders_ratio": "boulders",
            "ruins_ratio": "ruins",
            "structures_ratio": "structures",
            "soil_value": "soil",
        }
        feature_names = [feature_name_aliases.get(name, name) for name in feature_names]

        shap_array = np.array(getattr(shap_values, "values", shap_values))
        if shap_array.ndim == 3:
            class_index = 1 if shap_array.shape[2] > 1 else 0
            shap_vals = shap_array[0, :, class_index]
        elif shap_array.ndim == 2:
            shap_vals = shap_array[0]
        else:
            shap_vals = shap_array.reshape(-1)

        if shap_vals.shape[0] > features.shape[1]:
            shap_vals = shap_vals[:features.shape[1]]
        elif shap_vals.shape[0] < features.shape[1]:
            shap_vals = np.pad(shap_vals, (0, features.shape[1] - shap_vals.shape[0]), constant_values=0.0)

        top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_features = [feature_names[i] for i in top_indices]
        top_values = [float(shap_vals[i]) for i in top_indices]

        viz_order = np.argsort(np.abs(shap_vals))[::-1]
        viz_names = [feature_names[i] for i in viz_order]
        viz_vals = [float(shap_vals[i]) for i in viz_order]
        viz_colors = ["#ef4444" if v > 0 else "#22c55e" for v in viz_vals]

        shap_fig, ax = plt.subplots(figsize=(8, 4.2))
        y_pos = np.arange(len(viz_names))
        ax.barh(y_pos, viz_vals, color=viz_colors, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(viz_names)
        ax.invert_yaxis()
        ax.axvline(0, color="#94a3b8", linewidth=1)
        ax.set_xlabel("SHAP contribution")
        ax.set_title("Feature impact on erosion prediction")
        shap_fig.tight_layout()
    except Exception as e:
        shap_error = str(e)

    st.markdown(
        f"""
    <div style='text-align:center;margin-top:20px'>
        <h1 style='color:{risk_color};font-size:70px;margin-bottom:0'>
            {erosion_prob*100:.1f}%
        </h1>
        <p style='font-size:20px;color:{risk_color};margin-top:0'>
            {risk_zone} EROSION RISK
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.progress(int(erosion_prob * 100))

    st.markdown("### 🧠 Insight")
    st.markdown(
        f"""
    <div style="
        padding:15px;
        border-radius:12px;
        background:#111;
        border-left:4px solid {risk_color};
    ">
    {ai_explanation}
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("🖼️ Visual Analysis"):
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", width="stretch")
        col2.image(detection_img, caption="Detection", width="stretch")

        col3, col4 = st.columns(2)
        col3.image(seg_overlay, caption="Segmentation", width="stretch")
        col4.image(combined, caption="Combined", width="stretch")

        st.image(heat_overlay, caption="Erosion Heatmap", width="stretch")

    with st.expander("📊 Terrain Metrics"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Vegetation", f"{vegetation_ratio:.2f}")
        col2.metric("Slope", f"{slope:.2f}")
        col3.metric("Rainfall", f"{rainfall:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Elevation", f"{elevation:.2f}")
        col5.metric("Soil", soil_type.capitalize())

        st.caption(f"Coordinates: {lat:.4f}, {lon:.4f} ({location_source})")

    with st.expander("🔍 Model Explanation (Advanced)"):
        if top_features and top_values:
            for feature_name, feature_value in zip(top_features, top_values):
                if feature_value < 0:
                    st.write(f"🟢 {feature_name} reduces erosion")
                else:
                    st.write(f"🔴 {feature_name} increases erosion")
        elif shap_error:
            st.info(f"SHAP factors unavailable: {shap_error}")
        else:
            st.info("SHAP factors unavailable for this model output.")

        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)

    report_lines = [
        "Archaeological Site Analysis Report",
        "",
        f"Erosion Probability: {erosion_prob * 100:.2f}%",
        f"Risk Zone: {risk_zone}",
        f"Vegetation Ratio: {vegetation_ratio:.2f}",
        f"Slope: {slope:.2f}",
        f"Rainfall: {rainfall:.2f}",
        f"Elevation: {elevation:.2f}",
        f"Soil: {soil_type.capitalize()}",
        f"Boulders Ratio: {boulders_ratio:.4f}",
        f"Ruins Ratio: {ruins_ratio:.4f}",
        f"Structures Ratio: {structures_ratio:.4f}",
        f"Coordinates: {lat:.4f}, {lon:.4f}",
        f"Location Source: {location_source}",
        "",
        "AI Insight:",
        ai_explanation,
    ]
    report_content = "\n".join(report_lines)

    st.download_button(
        "📄 Download Report",
        data=report_content,
        file_name="report.txt",
        mime="text/plain",
    )
            