# ============================================================
# modules/image_processing.py — Core image analysis utilities
# ============================================================
import cv2
import numpy as np
import streamlit as st


def run_detection(img_bgr: np.ndarray, model, mode: str, confidence: int = 40):
    """Run YOLO inference or return a demo overlay if no model is loaded."""
    if model is None:
        rgb     = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        cv2.rectangle(overlay, (5, 5), (450, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, rgb, 0.5, 0, rgb)
        cv2.putText(rgb, "Demo Mode — No model loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 180, 120), 2)
        return rgb, []
    try:
        results = model(img_bgr, conf=max(0.05, min(0.95, confidence / 100.0)), verbose=False)
        dets, annotated = [], img_bgr
        for r in results:
            annotated = r.plot()
            if r.boxes is None:
                continue
            for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                dets.append({
                    "label": r.names[int(cls)],
                    "conf":  round(float(c), 3),
                    "bbox":  [x1, y1, x2, y2],
                    "cx":    (x1 + x2) // 2,
                    "cy":    (y1 + y2) // 2,
                })
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), dets
    except Exception as e:
        st.error(f"Detection failed: {e}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), []


def compute_vari(img_rgb: np.ndarray) -> np.ndarray:
    f    = img_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom   = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    return np.clip((g - r) / denom, -1, 1)


def colorise_vari(vari: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        cv2.applyColorMap(((vari + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )


def segment_vegetation(vari: np.ndarray):
    seg = np.zeros((*vari.shape, 3), dtype=np.uint8)
    layers = {
        "Very Dense": (vari > 0.7,                    (0, 100, 0)),
        "Dense":      ((vari > 0.5) & (vari <= 0.7),  (76, 175, 80)),
        "Moderate":   ((vari > 0.35) & (vari <= 0.5), (255, 193, 7)),
        "Sparse":     ((vari > 0.2) & (vari <= 0.35), (255, 152, 0)),
        "Bare Soil":  (vari <= 0.2,                   (161, 136, 127)),
    }
    total, cov = vari.size, {}
    for lbl, (mask, col) in layers.items():
        seg[mask] = col
        cov[lbl]  = round(mask.sum() / total * 100, 1)
    return seg, cov


def predict_erosion_score(model, feat_names, slope: float,
                          elevation: float, ndvi: float) -> float:
    if model is None:
        return float(np.clip(
            0.50 * np.clip(slope / 50, 0, 1) +
            0.35 * np.clip(1 - ((ndvi + 1) / 2), 0, 1) +
            0.15 * np.clip(elevation / 2000, 0, 1),
            0, 1,
        ))
    row = {f: 0.0 for f in feat_names}
    row.update({"slope": slope, "elevation": elevation, "ndvi": ndvi,
                "twi": 5.0, "tex_var": 50.0, "dist_water": 100.0, "curvature": 0.0})
    X = np.array([[row[f] for f in feat_names]])
    try:
        n = getattr(model, "n_features_in_", X.shape[1])
        return float(np.clip(model.predict(X[:, :n])[0], 0, 1))
    except Exception:
        return predict_erosion_score(None, feat_names, slope, elevation, ndvi)


def auto_detect_terrain(img_rgb: np.ndarray) -> dict:
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad  = np.sqrt(sx ** 2 + sy ** 2)
    slope = float(np.clip(
        (grad.mean() / (float(np.percentile(grad, 95)) or 1.0)) * 50.0, 0, 50
    ))
    bright = float(gray.mean()) / 255.0
    tex    = float(np.clip(cv2.Laplacian(gray, cv2.CV_32F).var() / 2000.0, 0, 1))
    dark   = float((gray < 60).sum()) / gray.size
    blue_d = float(np.clip(
        img_rgb[:, :, 2].astype(np.float32).mean() / 255.0 -
        img_rgb[:, :, 0].astype(np.float32).mean() / 255.0,
        0, 1,
    ))
    elev = float(np.clip(
        (0.30 * bright + 0.30 * tex + 0.25 * dark + 0.15 * blue_d) * 2000.0, 0, 2000
    ))
    conf = float(np.clip(float(gray.std()) / 128.0, 0.1, 1.0))
    return {"slope": round(slope, 1), "elevation": round(elev, 0), "confidence": round(conf, 2)}