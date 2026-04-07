# ============================================================
# modules/deforestation.py — Digital deforestation pipeline
# ============================================================
import cv2
import numpy as np
from typing import Tuple


def generate_vegetation_mask(img_rgb: np.ndarray,
                              vari_threshold: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    f    = img_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom   = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    vari    = np.clip((g - r) / denom, -1.0, 1.0)
    return vari > vari_threshold, vari


def remove_vegetation(img_rgb: np.ndarray, mask: np.ndarray,
                      intensity: float = 0.75) -> np.ndarray:
    out   = img_rgb.astype(np.float32).copy()
    earth = np.array([160.0, 130.0, 100.0], dtype=np.float32)

    for c in range(3):
        ch = out[:, :, c]
        ch[mask]    = ch[mask] * (1.0 - intensity) + earth[c] * intensity
        out[:, :, c] = ch

    green_excess = np.maximum(0.0, out[:, :, 1] - 0.5 * (out[:, :, 0] + out[:, :, 2]))
    suppress     = mask.astype(np.float32) * intensity
    out[:, :, 1] = out[:, :, 1] - green_excess * suppress
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_ground_features(img_rgb: np.ndarray, mask: np.ndarray,
                             clahe_clip: float = 3.0,
                             edge_strength: float = 0.6) -> np.ndarray:
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L_sharp = cv2.addWeighted(clahe.apply(L), 1.6,
                              cv2.GaussianBlur(clahe.apply(L), (0, 0), 3), -0.6, 0)
    enhanced = cv2.cvtColor(cv2.merge([L_sharp, A, B]), cv2.COLOR_LAB2RGB)

    gray_e   = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges    = cv2.Canny(gray_e, 40, 120)
    edge_col = np.array([255, 210, 80], dtype=np.float32)
    edge_mask = (edges > 0) & (~mask)

    ground = enhanced.astype(np.float32)
    for c in range(3):
        ground[:, :, c] = np.where(
            edge_mask,
            ground[:, :, c] * (1 - edge_strength) + edge_col[c] * edge_strength,
            ground[:, :, c],
        )
    ground[mask] = ground[mask] * 0.45
    return np.clip(ground, 0, 255).astype(np.uint8)


def detect_hidden_patterns(img_rgb: np.ndarray, mask: np.ndarray,
                            deforested: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
    gray = cv2.cvtColor(deforested, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    def _norm(arr):
        return arr / (arr.max() + 1e-6)

    lap       = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    tex_norm  = _norm(cv2.GaussianBlur(lap, (0, 0), max(h, w) // 60 or 5))

    edges     = cv2.Canny(gray.astype(np.uint8), 35, 110).astype(np.float32) / 255.0
    edge_norm = _norm(cv2.GaussianBlur(edges, (0, 0), max(h, w) // 50 or 5))

    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    lin_norm  = _norm(cv2.GaussianBlur(
        (sx + sy) / (np.abs(sx - sy) + sx + sy + 1e-6),
        (0, 0), max(h, w) // 55 or 5,
    ))

    veg_smooth = cv2.GaussianBlur((~mask).astype(np.float32),
                                  (0, 0), max(h, w) // 40 or 8)

    anomaly = (0.30 * tex_norm + 0.30 * edge_norm + 0.20 * lin_norm + 0.20 * veg_smooth)
    anomaly = _norm(cv2.GaussianBlur(anomaly, (0, 0), max(h, w) // 35 or 10))

    heatmap  = cv2.cvtColor(
        cv2.applyColorMap((anomaly * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    blended  = cv2.addWeighted(deforested, 0.50, heatmap, 0.50, 0)

    hot_mask    = anomaly >= float(np.percentile(anomaly, 88))
    hot_pct     = round(hot_mask.sum() / (h * w) * 100, 1)
    n_labels, _ = cv2.connectedComponents((hot_mask * 255).astype(np.uint8))

    stats = {
        "hotspot_pct":  hot_pct,
        "struct_count": max(0, n_labels - 1),
        "mean_anomaly": round(float(anomaly.mean()), 4),
        "peak_anomaly": round(float(anomaly.max()), 4),
        "veg_coverage": round(mask.sum() / (h * w) * 100, 1),
    }
    return blended, stats, anomaly


def build_vegetation_mask_visual(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask]  = (60, 160, 60)
    vis[~mask] = (40, 35, 30)
    return vis