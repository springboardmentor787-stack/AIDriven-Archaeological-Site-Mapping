import argparse
import base64
import json
import os
import sys
import traceback

import cv2
import joblib
import numpy as np
import shap
import torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Ensure project root modules (e.g., terrain_model) are importable when this script is
# executed from Next.js API routes.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from terrain_model.real_feature_extractor import extract_real_features
from terrain_model.ai_explainer_groq import generate_groq_explanation_with_status


_SHAP_EXPLAINER = None


def _resolve_segmentation_checkpoint():
    candidates = [
        os.path.join(ROOT, "models", "deeplab_model.pth"),
        os.path.join(ROOT, "runs", "segmentation", "deeplab_model_best.pth"),
        os.path.join(ROOT, "deeplab_model.pth"),
    ]

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("No segmentation checkpoint found.")

    return max(existing, key=os.path.getmtime)


def _resolve_erosion_model_path():
    candidates = [
        os.path.join(ROOT, "models", "erosion_model.pkl"),
        os.path.join(ROOT, "erosion_model.pkl"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No erosion model checkpoint found.")


def _load_models():
    yolo = YOLO(os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt"))

    seg = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6,
    )
    seg.load_state_dict(torch.load(_resolve_segmentation_checkpoint(), map_location="cpu"))
    seg.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = seg.to(device)

    erosion = joblib.load(_resolve_erosion_model_path())
    return yolo, seg, erosion, device


YOLO_MODEL, SEG_MODEL, EROSION_MODEL, DEVICE = _load_models()

CLASS_COLORS = {
    "boulders": (0, 255, 255),
    "others": (200, 200, 200),
    "ruins": (255, 0, 0),
    "structures": (0, 0, 255),
    "vegetation": (0, 255, 0),
}

SEG_COLORS = {
    0: [0, 0, 0],
    1: [0, 255, 255],
    2: [200, 200, 200],
    3: [255, 0, 0],
    4: [0, 0, 255],
    5: [0, 255, 0],
}

CLASS_ID_TO_NAME = {
    1: "boulders",
    2: "others",
    3: "ruins",
    4: "structures",
    5: "vegetation",
}


def _image_to_data_url(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode output image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_erosion_features(
    erosion_model,
    slope,
    vegetation_ratio,
    elevation,
    boulders_ratio,
    ruins_ratio,
    structures_ratio,
    rainfall=None,
    soil_value=None,
):
    expected_features = getattr(erosion_model, "n_features_in_", None)

    if expected_features == 3:
        return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3

    if expected_features == 4:
        return np.array([[slope, vegetation_ratio, ruins_ratio, elevation]], dtype=np.float32), 4

    if expected_features == 5:
        rainfall = rainfall if rainfall is not None else 0.0
        soil_value = soil_value if soil_value is not None else 2
        return np.array([[slope, vegetation_ratio, elevation, rainfall, soil_value]], dtype=np.float32), 5

    if expected_features == 6:
        return np.array(
            [[slope, vegetation_ratio, elevation, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32,
        ), 6

    if expected_features == 8:
        rainfall = rainfall if rainfall is not None else 0.0
        soil_value = soil_value if soil_value is not None else 2
        return np.array(
            [[slope, vegetation_ratio, elevation, rainfall, soil_value, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32,
        ), 8

    return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3


def _compute_top_shap(erosion_model, features, feature_mode):
    global _SHAP_EXPLAINER

    feature_names = list(getattr(erosion_model, "feature_names_in_", []))
    if len(feature_names) != features.shape[1]:
        feature_names_by_mode = {
            3: ["slope", "vegetation", "elevation"],
            4: ["slope", "vegetation", "ruins", "elevation"],
            5: ["slope", "vegetation", "elevation", "rainfall", "soil"],
            6: ["slope", "vegetation", "elevation", "boulders", "ruins", "structures"],
            8: ["slope", "vegetation", "elevation", "rainfall", "soil", "boulders", "ruins", "structures"],
        }
        feature_names = feature_names_by_mode.get(feature_mode, ["f" + str(i) for i in range(features.shape[1])])

    alias = {
        "vegetation_ratio": "vegetation",
        "boulders_ratio": "boulders",
        "ruins_ratio": "ruins",
        "structures_ratio": "structures",
        "soil_value": "soil",
    }
    feature_names = [alias.get(name, name) for name in feature_names]

    if _SHAP_EXPLAINER is None:
        _SHAP_EXPLAINER = shap.Explainer(erosion_model)

    shap_values = _SHAP_EXPLAINER(features)
    shap_array = np.array(getattr(shap_values, "values", shap_values))

    if shap_array.ndim == 3:
        class_index = 1 if shap_array.shape[2] > 1 else 0
        shap_vals = shap_array[0, :, class_index]
    elif shap_array.ndim == 2:
        shap_vals = shap_array[0]
    else:
        shap_vals = shap_array.reshape(-1)

    if shap_vals.shape[0] > features.shape[1]:
        shap_vals = shap_vals[: features.shape[1]]
    elif shap_vals.shape[0] < features.shape[1]:
        shap_vals = np.pad(shap_vals, (0, features.shape[1] - shap_vals.shape[0]), constant_values=0.0)

    top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
    return [{"feature": feature_names[i], "value": float(shap_vals[i])} for i in top_indices]


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_default_explanation(probability, slope, vegetation_ratio, rainfall):
    if probability >= 0.7:
        return (
            "High erosion risk is driven by terrain instability indicators. "
            f"Slope ({slope:.2f}) and rainfall ({rainfall:.2f}) suggest strong runoff pressure, "
            "while vegetation cover appears insufficient for full stabilization."
        )
    if probability >= 0.3:
        return (
            "Moderate erosion risk detected. Mixed indicators suggest partial stability with some vulnerable zones. "
            f"Vegetation ratio is {vegetation_ratio:.2f}, so targeted reinforcement is recommended."
        )
    return (
        "Low erosion risk detected. Current terrain indicators suggest generally stable conditions with lower immediate risk."
    )


def predict(
    image_path,
    lat,
    lon,
    api_key=None,
    confidence=0.25,
    class_visibility=None,
    use_ai_insight=False,
):
    if class_visibility is None:
        class_visibility = {
            "vegetation": True,
            "ruins": True,
            "structures": True,
            "boulders": True,
            "others": True,
        }

    conf_value = float(max(0.05, min(0.95, confidence)))

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Unable to read image")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_img = image.copy()
    yolo_results = YOLO_MODEL.predict(image, conf=conf_value, imgsz=640, verbose=False)
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = YOLO_MODEL.names[cls]
        if not class_visibility.get(label, True):
            continue
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(detection_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            detection_img,
            f"{label} {conf:.2f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    resized = cv2.resize(image, (512, 512))
    tensor = torch.tensor(resized.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = SEG_MODEL(tensor)
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))

    seg_vis = image.copy()
    for class_id, color in SEG_COLORS.items():
        if class_id == 0:
            continue
        class_name = CLASS_ID_TO_NAME[class_id]
        if class_name and class_visibility.get(class_name, True):
            seg_vis[mask == class_id] = color

    seg_overlay = cv2.addWeighted(image, 0.85, seg_vis, 0.15, 0)

    heatmap = np.zeros_like(image)
    heatmap[mask == 5] = [0, 0, 255]
    heatmap[mask == 3] = [255, 0, 0]
    heat_overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    combined = seg_overlay.copy()
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = YOLO_MODEL.names[cls]
        if not class_visibility.get(label, True):
            continue
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            combined,
            f"{label} {conf:.2f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    total_pixels = mask.size
    boulders_ratio = float(np.sum(mask == 1) / total_pixels)
    vegetation_ratio = float(np.sum(mask == 5) / total_pixels)
    ruins_ratio = float(np.sum(mask == 3) / total_pixels)
    structures_ratio = float(np.sum(mask == 4) / total_pixels)

    features_dict = extract_real_features(lat, lon, vegetation_ratio)
    slope = float(features_dict["slope"])
    rainfall = float(features_dict["rainfall"])
    soil = int(features_dict["soil"])
    elevation = float(features_dict["elevation"])

    features, feature_mode = build_erosion_features(
        EROSION_MODEL,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil,
    )

    if hasattr(EROSION_MODEL, "predict_proba"):
        probability = float(EROSION_MODEL.predict_proba(features)[0][1])
    else:
        raw = float(EROSION_MODEL.predict(features)[0])
        probability = max(0.0, min(1.0, raw))

    soil_map = {1: "sandy", 2: "loam", 3: "clay"}
    soil_type = soil_map.get(soil, "loam")

    ai_features = {
        "slope": slope,
        "vegetation": vegetation_ratio,
        "rainfall": rainfall,
        "soil": soil_type,
        "boulders": boulders_ratio,
        "ruins": ruins_ratio,
        "structures": structures_ratio,
    }
    insight_status = None
    if use_ai_insight:
        ai_explanation, insight_mode, insight_status = generate_groq_explanation_with_status(
            ai_features,
            probability,
            api_key=api_key or None,
        )
    else:
        ai_explanation = _build_default_explanation(probability, slope, vegetation_ratio, rainfall)
        insight_mode = "default"

    try:
        top_shap = _compute_top_shap(EROSION_MODEL, features, feature_mode)
    except Exception:
        top_shap = []

    return {
        "probability": probability,
        "riskLabel": "LOW" if probability < 0.3 else "MODERATE" if probability < 0.7 else "HIGH",
        "explanation": ai_explanation,
        "insightMode": insight_mode,
        "insightStatus": insight_status,
        "metrics": {
            "vegetation": vegetation_ratio,
            "slope": slope,
            "rainfall": rainfall,
            "elevation": elevation,
            "soil": soil_type,
            "boulders": boulders_ratio,
            "ruins": ruins_ratio,
            "structures": structures_ratio,
            "lat": lat,
            "lon": lon,
        },
        "shap": top_shap,
        "images": {
            "original": _image_to_data_url(image),
            "detection": _image_to_data_url(detection_img),
            "segmentation": _image_to_data_url(seg_overlay),
            "combined": _image_to_data_url(combined),
            "heatmap": _image_to_data_url(heat_overlay),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--confidence", default="0.25")
    parser.add_argument("--show-vegetation", default="true")
    parser.add_argument("--show-ruins", default="true")
    parser.add_argument("--show-structures", default="true")
    parser.add_argument("--show-boulders", default="true")
    parser.add_argument("--show-others", default="true")
    parser.add_argument("--use-ai-insight", default="false")
    args = parser.parse_args()

    try:
        payload = predict(
            args.image,
            args.lat,
            args.lon,
            args.api_key,
            confidence=float(args.confidence),
            class_visibility={
                "vegetation": _to_bool(args.show_vegetation),
                "ruins": _to_bool(args.show_ruins),
                "structures": _to_bool(args.show_structures),
                "boulders": _to_bool(args.show_boulders),
                "others": _to_bool(args.show_others),
            },
            use_ai_insight=_to_bool(args.use_ai_insight),
        )
        print(json.dumps({"ok": True, "data": payload}))
    except Exception as exc:
        error = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }
        print(json.dumps(error))


if __name__ == "__main__":
    main()
