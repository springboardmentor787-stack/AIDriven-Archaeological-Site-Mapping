# ============================================================
# utils/export.py — KML / KMZ export helpers
# ============================================================
import io
import zipfile
from typing import List


def build_kml(lat: float, lon: float, detections: List[dict], risk: float) -> str:
    marks = "".join(
        f"\n  <Placemark><n>{d['label']} #{i + 1}</n>"
        f"<description>Confidence: {d['conf']:.2%}</description>"
        f"<Point><coordinates>{lon + i * 0.0001},{lat + i * 0.0001},0</coordinates></Point></Placemark>"
        for i, d in enumerate(detections)
    )
    lbl = "Low" if risk < 0.33 else ("Moderate" if risk < 0.66 else "High")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        '  <n>Archaeological Site</n>\n'
        f'  <Placemark><n>Site Origin</n>\n'
        f'    <description>Erosion Risk: {risk:.2%} ({lbl})</description>\n'
        f'    <Point><coordinates>{lon},{lat},0</coordinates></Point>\n'
        f'  </Placemark>{marks}\n</Document></kml>'
    )


def build_kmz(kml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return buf.getvalue()