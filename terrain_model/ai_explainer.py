import os
from typing import Any, Dict


GEMINI_MODEL_CANDIDATES = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _soil_to_text(soil: Any) -> str:
    if isinstance(soil, str):
        return soil
    mapping = {1: "sandy", 2: "loam", 3: "clay"}
    try:
        return mapping.get(int(soil), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def build_prompt(features: Dict[str, Any], erosion_prob: float) -> str:
    return f"""
You are an environmental expert AI.

Analyze the terrain data below:

Slope: {_safe_float(features.get('slope')):.2f}
Vegetation: {_safe_float(features.get('vegetation')):.2f}
Rainfall: {_safe_float(features.get('rainfall')):.2f}
Soil Type: {_soil_to_text(features.get('soil'))}
Boulders: {_safe_float(features.get('boulders')):.2f}
Ruins: {_safe_float(features.get('ruins')):.2f}
Structures: {_safe_float(features.get('structures')):.2f}

Erosion Probability: {_safe_float(erosion_prob) * 100:.2f}%

Explain clearly:
- Why erosion risk is high or low
- What factors contributed most
- Keep it short (3-4 lines)
""".strip()


def generate_ai_explanation(features: Dict[str, Any], erosion_prob: float) -> str:
    slope = _safe_float(features.get("slope"))
    vegetation = _safe_float(features.get("vegetation"))
    rainfall = _safe_float(features.get("rainfall"))
    soil = _soil_to_text(features.get("soil"))

    reasons = []
    if rainfall > 100:
        reasons.append("high rainfall")
    if slope > 20:
        reasons.append("steep slope")
    if vegetation < 0.3:
        reasons.append("low vegetation cover")
    if soil == "sandy":
        reasons.append("erosion-prone sandy soil")

    risk = "high" if erosion_prob >= 0.7 else "moderate" if erosion_prob >= 0.4 else "low"
    reason_text = ", ".join(reasons[:3]) if reasons else "balanced terrain indicators"

    return (
        f"Estimated erosion risk is {risk} ({erosion_prob * 100:.1f}%). "
        f"The strongest contributors are {reason_text}. "
        "Use vegetation restoration, runoff control, and periodic monitoring to reduce future erosion."
    )


def generate_gemini_explanation(features: Dict[str, Any], erosion_prob: float, api_key: str | None = None) -> str:
    prompt = build_prompt(features, erosion_prob)

    return generate_gemini_from_prompt(
        prompt,
        fallback_text=generate_ai_explanation(features, erosion_prob),
        api_key=api_key,
    )


def generate_gemini_from_prompt(prompt: str, fallback_text: str, api_key: str | None = None) -> str:
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return f"⚠️ Gemini key missing. Showing local fallback explanation.\n\n{fallback_text}"

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        last_error_name = "UnknownError"
        last_error_text = ""

        for model_name in GEMINI_MODEL_CANDIDATES:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = getattr(response, "text", "")
                if text and text.strip():
                    return text.strip()
            except Exception as model_error:
                last_error_name = type(model_error).__name__
                last_error_text = str(model_error)
                continue

        if "quota" in last_error_text.lower() or last_error_name == "ResourceExhausted":
            return f"⚠️ Gemini quota exceeded on all configured models. Showing local fallback explanation.\n\n{fallback_text}"

        return f"⚠️ Gemini unavailable ({last_error_name}) on all configured models. Showing local fallback explanation.\n\n{fallback_text}"
    except Exception as e:
        error_text = str(e)
        error_name = type(e).__name__

        if "quota" in error_text.lower() or error_name == "ResourceExhausted":
            return f"⚠️ Gemini quota exceeded. Showing local fallback explanation.\n\n{fallback_text}"

        return f"⚠️ Gemini unavailable ({error_name}). Showing local fallback explanation.\n\n{fallback_text}"
