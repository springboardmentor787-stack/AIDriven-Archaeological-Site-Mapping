# ============================================================
# utils/geocoding.py — Multi-engine location geocoding
# ============================================================
import re
import requests
from config.settings import GEOCODE_HEADERS


def _try_nominatim(query: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1,
                    "addressdetails": 1, "accept-language": "en"},
            headers=GEOCODE_HEADERS, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            best  = data[0]
            short = ", ".join(best.get("display_name", query).split(", ")[:3])
            return float(best["lat"]), float(best["lon"]), short
    except Exception:
        pass
    return None, None, ""


def _try_nominatim_structured(village: str, state: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"village": village.strip(), "state": state.strip(),
                    "country": "India", "format": "json", "limit": 1,
                    "accept-language": "en"},
            headers=GEOCODE_HEADERS, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            best  = data[0]
            short = ", ".join(best.get("display_name", village).split(", ")[:3])
            return float(best["lat"]), float(best["lon"]), short
    except Exception:
        pass
    return None, None, ""


def _try_photon(query: str):
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": query, "limit": 1, "lang": "en"},
            headers=GEOCODE_HEADERS, timeout=10,
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
        if feats:
            props  = feats[0].get("properties", {})
            coords = feats[0]["geometry"]["coordinates"]
            parts  = [props.get(k, "") for k in ("name", "state", "country") if props.get(k)]
            return float(coords[1]), float(coords[0]), ", ".join(parts) or query
    except Exception:
        pass
    return None, None, ""


def geocode_location(location_name: str):
    """
    Try multiple geocoding engines and return (lat, lon, message).
    Returns (None, None, error_message) if all attempts fail.
    """
    if not location_name.strip():
        return None, None, "Please enter a location name."

    raw = re.sub(r"\s+", " ", re.sub(r",\s*", ", ", location_name.strip()))

    for fn, arg in [
        (_try_nominatim, raw),
        (_try_nominatim, "" if "india" in raw.lower() else f"{raw}, India"),
        (_try_photon,    raw),
        (_try_photon,    f"{raw} India"),
    ]:
        if not arg:
            continue
        lat, lon, disp = fn(arg)
        if lat is not None:
            return lat, lon, f"Located: {disp}"

    tokens = [p.strip() for p in raw.split(",") if p.strip()]
    if len(tokens) >= 2:
        lat, lon, disp = _try_nominatim_structured(tokens[0], tokens[-1])
        if lat is not None:
            return lat, lon, f"Located: {disp}"

    first = tokens[0] if tokens else raw.split()[0]
    for fn, arg in [(_try_nominatim, f"{first}, India"), (_try_photon, f"{first} India")]:
        lat, lon, disp = fn(arg)
        if lat is not None:
            return lat, lon, f"Located: {disp} (nearest for '{first}')"

    skip = {"india", "maharashtra", "karnataka", "tamilnadu", "tamil nadu",
            "gujarat", "rajasthan", "uttarpradesh", "uttar pradesh", "madhya pradesh"}
    for t in tokens:
        if len(t) > 3 and t.lower() not in skip:
            lat, lon, disp = _try_photon(f"{t} India")
            if lat is not None:
                return lat, lon, f"Located: {disp} (matched on '{t}')"

    return None, None, (
        f"Could not locate '{location_name}'. "
        "Try adding district/state, e.g. 'Daimabad, Ahmednagar, Maharashtra, India', "
        "or enter coordinates manually."
    )