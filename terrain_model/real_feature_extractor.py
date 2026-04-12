import requests
import numpy as np
from functools import lru_cache


_HTTP = requests.Session()
_REQ_TIMEOUT = 2.0


def _fetch_json(url, timeout=_REQ_TIMEOUT):
    response = _HTTP.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()

# -------------------------------
# 1️⃣ Elevation API
# -------------------------------

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        res = _fetch_json(url)
        return res["results"][0]["elevation"]
    except:
        return 350  # fallback


def get_elevations(points):
    try:
        location_str = "|".join([f"{lat},{lon}" for lat, lon in points])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={location_str}"
        res = _fetch_json(url)
        vals = [item.get("elevation", 350) for item in res.get("results", [])]
        if len(vals) == len(points):
            return vals
    except:
        pass

    return [get_elevation(lat, lon) for lat, lon in points]


# -------------------------------
# 2️⃣ Rainfall (7-day average)
# -------------------------------

def get_avg_rainfall(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&past_days=7"
        res = _fetch_json(url)
        rain = res["daily"]["precipitation_sum"]
        return sum(rain) / len(rain)
    except:
        return 50  # fallback


# -------------------------------
# 3️⃣ Soil (SoilGrids API)
# -------------------------------

def get_soil_type(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&depth=0-5cm"
        res = _fetch_json(url)

        clay = res["properties"]["layers"][0]["depths"][0]["values"]["mean"]

        if clay > 40:
            return 3  # clay
        elif clay > 20:
            return 2  # loam
        else:
            return 1  # sandy
    except:
        return 2  # fallback


# -------------------------------
# 4️⃣ Slope (from elevation variation)
# -------------------------------

def get_slope(lat, lon):
    try:
        e1, e2, e3 = get_elevations([
            (lat, lon),
            (lat + 0.001, lon),
            (lat, lon + 0.001),
        ])

        dx = abs(e2 - e1)
        dy = abs(e3 - e1)

        slope = (dx + dy) / 2
        return slope
    except:
        return 10  # fallback


# -------------------------------
# 5️⃣ Combine all features
# -------------------------------

@lru_cache(maxsize=512)
def _extract_real_features_cached(lat_key, lon_key):
    elevation = get_elevation(lat_key, lon_key)
    rainfall = get_avg_rainfall(lat_key, lon_key)
    soil = get_soil_type(lat_key, lon_key)
    slope = get_slope(lat_key, lon_key)

    return {
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil": soil,
    }


def extract_real_features(lat, lon, vegetation_ratio):

    lat_key = round(float(lat), 4)
    lon_key = round(float(lon), 4)
    cached = _extract_real_features_cached(lat_key, lon_key)

    return {
        "slope": cached["slope"],
        "vegetation": vegetation_ratio,
        "elevation": cached["elevation"],
        "rainfall": cached["rainfall"],
        "soil": cached["soil"],
    }