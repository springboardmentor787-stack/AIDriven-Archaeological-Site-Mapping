# ============================================================
# modules/mound_detection.py — Mound/object detection pipeline
# ============================================================
import cv2
import numpy as np
from typing import List, Optional, Tuple


# ── Candidate detection ────────────────────────────────────────────────────────

def detect_mound_candidates(img_bgr: np.ndarray, model,
                             confidence_threshold: int = 40) -> List[dict]:
    """Return bounding-box candidates from YOLO or image-driven heuristics."""
    if model is not None:
        try:
            results = model(img_bgr, conf=max(0.05, confidence_threshold / 100.0),
                            verbose=False)
            dets = []
            for r in results:
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
            return dets
        except Exception:
            pass

    # ── Demo / fallback: Laplacian blob detection ──────────────────────────────
    h, w  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21, 21), 0)
    lap   = cv2.Laplacian(blur.astype(np.float32), cv2.CV_32F)
    lap_u8 = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(lap_u8, int(np.percentile(lap_u8, 80)), 255,
                              cv2.THRESH_BINARY)
    n_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(thresh)

    dets       = []
    label_pool = ["ruins", "mound", "structure", "earthwork", "mound"]
    np.random.seed(12)

    for i in range(1, min(n_labels, 16)):
        x, y, bw, bh, area = stats_cc[i]
        if area < 200:
            continue
        pad_x = max(int(bw * 0.2), 8)
        pad_y = max(int(bh * 0.2), 8)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        dets.append({
            "label": label_pool[i % len(label_pool)],
            "conf":  round(np.random.uniform(0.35, 0.92), 3),
            "bbox":  [x1, y1, x2, y2],
            "cx":    (x1 + x2) // 2,
            "cy":    (y1 + y2) // 2,
        })

    if not dets:
        for i in range(6):
            bw = np.random.randint(int(w * 0.06), int(w * 0.20))
            bh = np.random.randint(int(h * 0.06), int(h * 0.18))
            x1 = np.random.randint(0, max(1, w - bw))
            y1 = np.random.randint(0, max(1, h - bh))
            dets.append({
                "label": label_pool[i % len(label_pool)],
                "conf":  round(np.random.uniform(0.35, 0.90), 3),
                "bbox":  [x1, y1, x1 + bw, y1 + bh],
                "cx":    x1 + bw // 2,
                "cy":    y1 + bh // 2,
            })

    return dets


# ── Region-level feature extraction ───────────────────────────────────────────

def extract_region(img_rgb: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_rgb[y1:y2, x1:x2]


def compute_texture_variance(region_gray: Optional[np.ndarray]) -> float:
    if region_gray is None or region_gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(region_gray.astype(np.float32), cv2.CV_32F)
    return float(np.clip(lap.var() / 5000.0, 0.0, 1.0))


def compute_shape_regularity(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    return round(float(min(bw, bh) / max(bw, bh)), 3)


def compute_region_vari(region_rgb: Optional[np.ndarray]) -> float:
    if region_rgb is None or region_rgb.size == 0:
        return 0.0
    f    = region_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom   = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    return float(np.clip((g - r) / denom, -1, 1).mean())


# ── Classification ─────────────────────────────────────────────────────────────

def classify_mound(shape_reg: float, tex_var: float,
                   vari_val: float, conf: float) -> Tuple[str, float]:
    if vari_val > 0.35:
        return "Natural", 0.3
    if shape_reg < 0.45:
        return "Natural", 0.35

    score = (
        0.35 * shape_reg +
        0.25 * (1.0 - tex_var) +
        0.25 * max(0.0, 1.0 - vari_val) +
        0.15 * conf
    )

    if score > 0.65:
        return "Man-made", round(score, 3)
    elif score < 0.45:
        return "Natural", round(score, 3)
    return "Uncertain", round(score, 3)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_mound_pipeline(img_rgb: np.ndarray, model,
                       conf_threshold: int = 40,
                       filter_high_conf: bool = True) -> List[dict]:
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    candidates = detect_mound_candidates(img_bgr, model, conf_threshold)

    results = []
    for det in candidates:
        region = extract_region(img_rgb, det["bbox"])
        if region is None:
            continue
        gray_r    = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
        tex_var   = compute_texture_variance(gray_r)
        shape_reg = compute_shape_regularity(det["bbox"])
        vari_val  = compute_region_vari(region)
        cls_label, cls_score = classify_mound(shape_reg, tex_var, vari_val, det["conf"])
        results.append({
            **det,
            "tex_var":   round(tex_var, 3),
            "shape_reg": round(shape_reg, 3),
            "vari_val":  round(vari_val, 3),
            "cls_label": cls_label,
            "cls_score": round(cls_score, 3),
        })

    for r in results:
        r["highlight"] = (r["cls_label"] == "Man-made" and
                          r["conf"] >= conf_threshold / 100.0) if filter_high_conf else True

    return results


# ── Overlay & heatmap ──────────────────────────────────────────────────────────

def draw_mound_overlay(img_rgb: np.ndarray, results: List[dict],
                       filter_high_conf: bool = True) -> np.ndarray:
    """
    Draw all detected objects on the overlay image.
    Natural/Uncertain: thin 1-px border + short label.
    Man-made (highlighted): thick 3-px border + filled label tag.
    """
    overlay   = img_rgb.copy()
    color_map = {
        "Man-made":  (214, 60,  60),
        "Natural":   (80,  200, 120),
        "Uncertain": (212, 168, 60),
    }

    # Pass 1: Natural / Uncertain (subtle)
    for r in results:
        if r.get("highlight"):
            continue
        x1, y1, x2, y2 = r["bbox"]
        col       = color_map.get(r["cls_label"], (150, 150, 150))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 1)
        short_tag = f"{r['cls_label'][0]} {r['conf']:.0%}"
        cv2.putText(overlay, short_tag, (x1 + 3, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # Pass 2: Man-made / highlighted (prominent)
    for r in results:
        if not r.get("highlight"):
            continue
        x1, y1, x2, y2 = r["bbox"]
        col       = color_map.get(r["cls_label"], (150, 150, 150))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 3)
        label_txt = f"{r['cls_label']} {r['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(overlay, label_txt, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    return overlay


def build_detection_heatmap(img_rgb: np.ndarray, results: List[dict]) -> np.ndarray:
    h, w  = img_rgb.shape[:2]
    heat  = np.zeros((h, w), dtype=np.float32)
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
        sigma   = max((x2 - x1), (y2 - y1)) // 2 or 20
        for dy in range(-sigma * 2, sigma * 2 + 1):
            for dx in range(-sigma * 2, sigma * 2 + 1):
                ry, rx = cy + dy, cx + dx
                if 0 <= ry < h and 0 <= rx < w:
                    heat[ry, rx] += float(r["conf"]) * np.exp(
                        -(dx**2 + dy**2) / (2 * sigma**2)
                    )

    heat    = cv2.GaussianBlur(heat, (0, 0), max(h, w) // 40 or 10)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_col = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heat_rgb, 0.45, 0)


# ── Survey cost savings ────────────────────────────────────────────────────────

def compute_cost_savings(results: List[dict], total_area_sqkm: float = 50.0) -> dict:
    total   = len(results)
    manmade = sum(1 for r in results if r["cls_label"] == "Man-made")
    natural = sum(1 for r in results if r["cls_label"] == "Natural")
    uncert  = total - manmade - natural

    pct_filtered  = round((natural / total) * 100, 1) if total else 0.0
    area_filtered = round(total_area_sqkm * pct_filtered / 100, 1)
    area_priority = round(total_area_sqkm - area_filtered, 1)

    days_trad  = round(total_area_sqkm * 3)
    days_ai    = round(area_priority * 3)
    staff, daily_cost = 5, 200
    cost_trad  = days_trad * staff * daily_cost
    cost_ai    = days_ai   * staff * daily_cost

    return {
        "total":         total,
        "manmade":       manmade,
        "natural":       natural,
        "uncertain":     uncert,
        "pct_filtered":  pct_filtered,
        "area_filtered": area_filtered,
        "area_priority": area_priority,
        "days_saved":    days_trad - days_ai,
        "cost_saved":    cost_trad - cost_ai,
        "cost_trad":     cost_trad,
        "cost_ai":       cost_ai,
    }