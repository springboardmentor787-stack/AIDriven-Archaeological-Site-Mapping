# ============================================================
# config/styles.py — Global CSS injected via st.markdown
# ============================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Archivo+Narrow:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Archivo Narrow', sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-primary);
}

/* ===== SIDEBAR FIX: Always keep sidebar visible & collapse arrow always shown ===== */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
}

[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    transition: none !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: translateX(0) !important;
}

.css-1d391kg, section[data-testid="stSidebarContent"] {
    display: block !important;
    visibility: visible !important;
}
/* ===== END SIDEBAR FIX ===== */

.stDeployButton { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

.sidebar-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 300;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.sidebar-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--accent-dim), transparent);
    margin: 0.8rem 0 1rem;
}

.stTextInput input, .stNumberInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 7px 14px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    background: rgba(184,150,110,0.05) !important;
    transform: none !important;
    box-shadow: none !important;
}

section[data-testid="stMain"] .stButton > button {
    background: linear-gradient(135deg, var(--bg-elevated), var(--bg-secondary)) !important;
    border: 1px solid var(--accent-dim) !important;
    color: var(--accent) !important;
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 2px 20px rgba(184,150,110,0.08) !important;
}
section[data-testid="stMain"] .stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 4px 30px rgba(184,150,110,0.18) !important;
    transform: translateY(-1px) !important;
}

.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px;
    font-weight: 300;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 3px;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.title-bar {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.04em !important;
}
h2 { font-size: 22px !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 1rem !important; }
h3, h4 { font-size: 16px !important; color: var(--text-secondary) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 24px !important;
    color: var(--text-primary) !important;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px !important; }
.stAlert { border-radius: 3px !important; border-left-width: 3px !important;
           font-family: 'Archivo Narrow', sans-serif !important; font-size: 13px !important; }

.stProgress > div > div { background: var(--bg-card) !important; border-radius: 2px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-dim), var(--accent)) !important;
    border-radius: 2px !important;
}

.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-light) !important;
    border-radius: 4px !important;
}

.stCaption, small {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
}

.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    text-transform: none !important;
    letter-spacing: 0.05em !important;
}

.mound-card {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
    background: var(--bg-card);
}
.mound-card.manmade { border-left: 3px solid #d46b6b; }
.mound-card.natural { border-left: 3px solid #7ec899; }
.mound-card.uncertain { border-left: 3px solid #d4a84b; }
.mound-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.mound-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 17px;
    color: var(--text-primary);
}
</style>
"""