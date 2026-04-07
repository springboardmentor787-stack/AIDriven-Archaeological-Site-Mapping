# ============================================================
# config/settings.py — App-wide constants and theme definitions
# ============================================================

GROQ_API_KEY = "YOUR_API"

APP_TITLE   = "ArchAI — Archaeological Intelligence Platform"
APP_ICON    = "A"
APP_VERSION = "4.2"
APP_AUTHOR  = "Hari Krishanan M"

GEOCODE_HEADERS = {"User-Agent": "ArchAI-Dashboard/1.0"}

THEMES = {
    "Dark": {
        "--bg-primary":     "#08090d",
        "--bg-secondary":   "#0e1118",
        "--bg-card":        "#111520",
        "--bg-elevated":    "#161c2a",
        "--border":         "#1e2535",
        "--border-light":   "#252f45",
        "--accent":         "#b8966e",
        "--accent-dim":     "#7a5f42",
        "--text-primary":   "#d4cfc8",
        "--text-secondary": "#7a8399",
        "--text-muted":     "#4a5268",
        "--risk-low-fg":    "#7ec899",
        "--risk-mod-fg":    "#d4a84b",
        "--risk-high-fg":   "#d46b6b",
    },
    "Light": {
        "--bg-primary":     "#f5f7fa",
        "--bg-secondary":   "#ffffff",
        "--bg-card":        "#ffffff",
        "--bg-elevated":    "#eef1f7",
        "--border":         "#e2e8f0",
        "--border-light":   "#cbd5e0",
        "--accent":         "#8b6f47",
        "--accent-dim":     "#a07850",
        "--text-primary":   "#1a202c",
        "--text-secondary": "#4a5568",
        "--text-muted":     "#718096",
        "--risk-low-fg":    "#276749",
        "--risk-mod-fg":    "#975a16",
        "--risk-high-fg":   "#c53030",
    },
}

SESSION_DEFAULTS = [
    ("theme",         "Dark"),
    ("dets",          []),
    ("risk",          0.0),
    ("vari_mean",     0.0),
    ("coverage",      {}),
    ("lat",           20.5937),
    ("lon",           78.9629),
    ("location_name", ""),
    ("geo_msg",       ""),
    ("auto_terrain",  {"slope": 0.0, "elevation": 0.0, "confidence": 0.0}),
    ("mound_results", []),
    ("deforest_results", None),
]