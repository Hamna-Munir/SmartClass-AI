# Week 2 — SmartClass AI · Console Edition
# Developer: Hamna Munir

import os
import math
import time
import base64
import tempfile

import cv2
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

from modules.emotion_detector  import detect_emotion
from modules.engagement_scorer import EngagementScorer
from modules.report_generator  import generate_class_report
from utils.logger              import init_log, log_entry, read_log, get_session_summary
from modules.face_tracker import analyze_classroom, get_class_summary

st.set_page_config(
    page_title="SmartClass AI · Console",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_log()

if "alerts"     not in st.session_state: st.session_state.alerts     = []
if "scorers"    not in st.session_state: st.session_state.scorers    = {}
if "class_data" not in st.session_state: st.session_state.class_data = []
if "video_data" not in st.session_state: st.session_state.video_data = []
if "agent" not in st.session_state:
    from modules.agent import ClassroomAgent
    st.session_state.agent = ClassroomAgent()
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()

# Streamlit forbids writing directly to st.session_state["nav_page"] once
# the sidebar radio (key="nav_page") has been instantiated in a run. So
# "Open →" buttons elsewhere on the page write to this separate
# "nav_target" flag instead and call st.rerun(); on the NEXT run — before
# the sidebar (and therefore the radio) is created — we copy that flag
# into "nav_page", which is allowed since the widget doesn't exist yet
# this run. This is what actually makes those buttons navigate.
if "nav_target" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("nav_target")

# ══════════════════════════════════════════════════════════════════
# DESIGN TOKENS — strict Black / White / Yellow
# ══════════════════════════════════════════════════════════════════
INK        = "#000000"   # app background — pure black
PANEL      = "#111111"   # card background
PANEL_ALT  = "#1B1B1B"   # nested panel background
SIDEBAR_BG = "#000000"
BORDER     = "#333333"
BORDER_SOFT= "#242424"
TEXT       = "#FFFFFF"
MUTED      = "#B3B3B3"
FAINT      = "#7A7A7A"
GOLD       = "#FFD400"   # signature yellow accent
GOLD_DIM   = "#E0BB00"
GOLD_SOFT  = "rgba(255,212,0,0.14)"
GREEN      = "#3ED67F"   # kept only as a functional "on-track" signal
GREEN_SOFT = "rgba(62,214,127,0.12)"
AMBER      = "#FF9F1C"   # kept only as a functional "moderate" signal
AMBER_SOFT = "rgba(255,159,28,0.12)"
RED        = "#FF5C5C"   # kept only as a functional "alert" signal
RED_SOFT   = "rgba(255,92,92,0.14)"
TEAL       = GOLD        # no blue/teal in the palette — reuse accent yellow

# Charts render on a clean white surface for maximum readability,
# framed by the black/white/yellow shell around them.
CHART_BG   = "#FFFFFF"
CHART_TEXT = "#111111"
CHART_GRID = "#E6E6E6"

# Elevation system — layered shadows (soft ambient + crisp contact shadow
# + a hairline top highlight) instead of a single flat drop-shadow. This
# is what makes cards read as "lifted" surfaces rather than flat boxes.
SHADOW_XS  = "0 1px 2px rgba(0,0,0,0.45)"
SHADOW_SM  = "0 2px 8px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.03) inset"
SHADOW_MD  = "0 8px 24px rgba(0,0,0,0.40), 0 1px 0 rgba(255,255,255,0.04) inset"
SHADOW_LG  = "0 20px 48px rgba(0,0,0,0.50), 0 1px 0 rgba(255,255,255,0.05) inset"
SHADOW_GOLD= "0 10px 28px rgba(255,212,0,0.16)"
EASE       = "cubic-bezier(0.22, 1, 0.36, 1)"

EMOTION_COLORS = {
    "Happy":"#3ED67F","Neutral":"#8A8A8A","Sad":"#5C6BC0",
    "Angry":"#FF5C5C","Fear":"#FF9F1C","Surprise":"#FFD400","Disgust":"#C6577A"
}
EMOJI_MAP = {
    "Happy":"😊","Sad":"😢","Angry":"😠",
    "Neutral":"😐","Fear":"😨","Surprise":"😲","Disgust":"🤢"
}

# ══════════════════════════════════════════════════════════════════
# CSS — INSTRUMENT PANEL THEME
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,900&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body {{ font-family:'Inter',sans-serif !important; }}
h1,h2,h3,.smc-display {{ font-family:'Fraunces','Inter',serif !important; }}
.smc-mono {{ font-family:'JetBrains Mono',monospace !important; }}

.stApp, [data-testid="stAppViewContainer"] {{ background:{INK} !important; }}
[data-testid="stHeader"] {{
    background:{INK} !important;
    border-bottom:1px solid {BORDER_SOFT} !important;
    box-shadow:none !important;
}}
.smc-topbar {{
    height:5px; width:100%;
    background:linear-gradient(90deg, #000000 0%, {GOLD} 25%, #FFF3B0 50%, {GOLD} 75%, #000000 100%);
    background-size:200% 100%;
    box-shadow:0 0 18px rgba(255,212,0,0.55);
    margin-bottom:2px;
}}
.main .block-container {{ padding-top:22px !important; max-width:1400px; }}
p, span, label, div {{ color:{TEXT}; }}

/* ── MASTHEAD ── */
.smc-masthead {{
    display:flex; align-items:center; gap:20px;
    padding:6px 0 26px 0; margin-bottom:8px;
    border-bottom:1px solid {BORDER_SOFT};
}}
.smc-masthead-badge {{
    width:64px; height:64px; border-radius:16px; flex-shrink:0;
    background:linear-gradient(155deg, {GOLD} 0%, #B8860B 100%);
    display:flex; align-items:center; justify-content:center;
    font-size:30px; box-shadow:0 0 0 1px rgba(255,212,0,0.4), 0 10px 30px rgba(255,212,0,0.28);
}}
.smc-masthead-title {{
    font-family:'Fraunces',serif; font-weight:900; font-size:46px; line-height:1;
    letter-spacing:-0.01em; margin:0;
    background:linear-gradient(90deg, #FFFFFF 0%, {GOLD} 65%, {GOLD} 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}}
.smc-masthead-tag {{
    font-family:'JetBrains Mono',monospace; font-size:11.5px; letter-spacing:0.22em;
    color:{GOLD}; text-transform:uppercase; margin-top:6px; opacity:0.85;
}}

/* ── SIDEBAR — CONTROL PANEL ── */
[data-testid="stSidebar"] {{
    background:{SIDEBAR_BG} !important;
    border-right:1px solid {BORDER_SOFT} !important;
    box-shadow:8px 0 24px rgba(0,0,0,0.35) !important;
}}
[data-testid="stSidebar"] * {{ color:{MUTED} !important; }}
[data-testid="stSidebar"] .stRadio > div {{
    display:flex !important; flex-direction:column !important;
    gap:5px !important; padding:0 10px !important;
}}
[data-testid="stSidebar"] .stRadio input {{ display:none !important; }}
[data-testid="stSidebar"] .stRadio label > div:first-child {{ display:none !important; }}
[data-testid="stSidebar"] .stRadio label {{
    position:relative !important;
    display:flex !important; align-items:center !important;
    padding:11px 14px 11px 18px !important; border-radius:10px !important;
    font-size:13.5px !important; font-weight:600 !important;
    color:{MUTED} !important; cursor:pointer !important;
    background:{PANEL} !important;
    border:1px solid {BORDER_SOFT} !important;
    box-shadow:{SHADOW_XS} !important;
    transition:background 0.2s {EASE}, border-color 0.2s {EASE},
               color 0.2s {EASE}, transform 0.2s {EASE}, box-shadow 0.2s {EASE} !important;
    margin:0 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background:{PANEL_ALT} !important;
    border-color:rgba(255,212,0,0.35) !important;
    color:{TEXT} !important;
    transform:translateX(3px) !important;
    box-shadow:{SHADOW_SM} !important;
}}
[data-testid="stSidebar"] [aria-checked="true"] + label {{
    background:linear-gradient(90deg, {GOLD} 0%, {GOLD_DIM} 100%) !important;
    border-color:{GOLD} !important;
    color:{INK} !important; font-weight:800 !important;
    box-shadow:0 6px 20px rgba(255,212,0,0.38), 0 1px 0 rgba(255,255,255,0.3) inset !important;
    transform:translateX(2px) !important;
}}
[data-testid="stSidebar"] [aria-checked="true"] + label::before {{
    content:'' !important; position:absolute !important; left:-10px !important;
    top:50% !important; transform:translateY(-50%) !important;
    width:4px !important; height:60% !important; border-radius:0 4px 4px 0 !important;
    background:{GOLD} !important; box-shadow:0 0 10px rgba(255,212,0,0.7) !important;
}}

/* ── SEGMENTED CONTROL (main content radios) ── */
[data-testid="stAppViewContainer"] [data-testid="stRadio"] > div[role="radiogroup"] {{
    display:flex !important; gap:8px !important; flex-wrap:wrap !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] input {{ display:none !important; }}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label > div:first-child {{ display:none !important; }}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label {{
    background:{PANEL} !important; border:1.5px solid {BORDER} !important;
    border-radius:11px !important; padding:12px 22px !important;
    font-size:13.5px !important; font-weight:700 !important;
    color:{MUTED} !important; cursor:pointer !important; margin:0 !important;
    box-shadow:{SHADOW_XS} !important;
    transition:all 0.22s {EASE} !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label:hover {{
    border-color:{GOLD} !important; color:{TEXT} !important;
    transform:translateY(-2px) !important;
    box-shadow:{SHADOW_SM} !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] [aria-checked="true"] + label {{
    background:linear-gradient(135deg, {GOLD} 0%, {GOLD_DIM} 100%) !important;
    border-color:{GOLD} !important; color:{INK} !important;
    box-shadow:0 8px 22px rgba(255,212,0,0.4), 0 1px 0 rgba(255,255,255,0.3) inset !important;
    transform:translateY(-1px) !important;
}}

/* ── METRICS ── */
[data-testid="stMetric"] {{
    background:linear-gradient(160deg, {PANEL} 0%, {PANEL_ALT} 100%) !important;
    border:1px solid {BORDER} !important;
    border-top:3px solid {GOLD} !important;
    border-radius:14px !important;
    padding:18px 20px !important;
    box-shadow:{SHADOW_SM} !important;
    transition:box-shadow 0.25s {EASE}, transform 0.25s {EASE} !important;
}}
[data-testid="stMetric"]:hover {{
    box-shadow:{SHADOW_MD}, {SHADOW_GOLD} !important;
    transform:translateY(-3px) !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size:10.5px !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.12em !important;
    color:{FAINT} !important;
}}
[data-testid="stMetricValue"] {{
    font-family:'JetBrains Mono',monospace !important;
    font-size:30px !important; font-weight:800 !important; color:{TEXT} !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background:linear-gradient(135deg, {GOLD} 0%, {GOLD_DIM} 100%) !important;
    color:{INK} !important;
    border:none !important; border-radius:10px !important;
    font-size:13.5px !important; font-weight:800 !important;
    letter-spacing:0.03em !important;
    padding:12px 20px !important; width:100% !important;
    box-shadow:0 4px 16px rgba(255,212,0,0.3), 0 1px 0 rgba(255,255,255,0.35) inset !important;
    transition:all 0.22s {EASE} !important;
}}
.stButton > button:hover {{
    box-shadow:0 10px 28px rgba(255,212,0,0.45), 0 1px 0 rgba(255,255,255,0.35) inset !important;
    transform:translateY(-2px) scale(1.01) !important;
    filter:brightness(1.06) !important;
}}
.stButton > button:active {{
    transform:translateY(0) scale(0.99) !important;
    box-shadow:0 2px 8px rgba(255,212,0,0.35) !important;
}}
.stDownloadButton > button {{
    background:{PANEL} !important; color:{TEXT} !important;
    border:1.5px solid {GOLD} !important; border-radius:10px !important;
    font-weight:700 !important;
    box-shadow:{SHADOW_XS} !important;
    transition:all 0.22s {EASE} !important;
}}
.stDownloadButton > button:hover {{
    background:{GOLD_SOFT} !important; color:{GOLD} !important;
    transform:translateY(-2px) !important;
    box-shadow:{SHADOW_SM} !important;
}}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {{
    background:{PANEL_ALT} !important;
    border-radius:11px !important; padding:4px !important;
    border:1px solid {BORDER_SOFT} !important; gap:2px !important;
    box-shadow:0 2px 6px rgba(0,0,0,0.3) inset !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    border-radius:8px !important; font-size:13px !important;
    font-weight:600 !important; color:{MUTED} !important;
    border:none !important; padding:9px 16px !important;
    transition:all 0.2s {EASE} !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    color:{TEXT} !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background:{GOLD} !important; color:{INK} !important;
    box-shadow:0 4px 14px rgba(255,212,0,0.35), 0 1px 0 rgba(255,255,255,0.3) inset !important;
}}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top:16px !important; }}

/* ── ALERTS ── */
.stSuccess {{ background:{GREEN_SOFT} !important; border:1px solid rgba(62,214,127,0.35) !important; border-radius:10px !important; }}
.stWarning {{ background:{AMBER_SOFT} !important; border:1px solid rgba(255,159,28,0.35) !important; border-radius:10px !important; }}
.stError   {{ background:{RED_SOFT} !important;   border:1px solid rgba(255,92,92,0.35) !important;  border-radius:10px !important; }}
.stInfo    {{ background:rgba(255,212,0,0.10) !important; border:1px solid rgba(255,212,0,0.35) !important; border-radius:10px !important; }}
.stSuccess *, .stWarning *, .stError *, .stInfo * {{ color:{TEXT} !important; }}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {{ border-radius:10px !important; border:1px solid {BORDER} !important; }}

/* ── FILE UPLOADER / CAMERA ── */
[data-testid="stFileUploader"] {{
    background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%) !important;
    border:1.5px dashed {BORDER} !important;
    border-radius:14px !important; padding:20px !important;
    box-shadow:{SHADOW_XS} !important;
    transition:border-color 0.2s {EASE}, box-shadow 0.2s {EASE} !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color:rgba(255,212,0,0.4) !important;
    box-shadow:{SHADOW_SM} !important;
}}
[data-testid="stFileUploader"] section {{ background:transparent !important; }}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img {{ border-radius:12px !important; box-shadow:{SHADOW_SM} !important; }}
[data-testid="stTextInput"] input {{
    background:{PANEL} !important; color:{TEXT} !important; border:1px solid {BORDER} !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background:{PANEL} !important; border-color:{BORDER} !important; color:{TEXT} !important;
}}
hr {{ border-color:{BORDER_SOFT} !important; }}

/* ── HOVERABLE CARD — every card() lifts on hover, same language as
   the metric cards, so the whole UI feels consistently interactive ── */
.smc-card {{
    transition:transform 0.25s {EASE}, box-shadow 0.25s {EASE}, border-color 0.25s {EASE} !important;
}}
.smc-card:hover {{
    transform:translateY(-4px) !important;
    box-shadow:{SHADOW_MD}, 0 0 0 1px rgba(255,212,0,0.18) !important;
    border-color:rgba(255,212,0,0.3) !important;
}}

/* ── QUICK ACTION tiles ── */
.smc-qa {{
    background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%);
    border:1px solid {BORDER}; border-radius:14px;
    padding:18px 14px 14px; text-align:center;
    box-shadow:{SHADOW_XS};
    transition:transform 0.25s {EASE}, box-shadow 0.25s {EASE}, border-color 0.25s {EASE};
}}
.smc-qa:hover {{
    transform:translateY(-4px) !important;
    box-shadow:{SHADOW_MD}, 0 0 0 1px rgba(255,212,0,0.25) !important;
    border-color:{GOLD} !important;
}}
.smc-qa .smc-qa-icon {{ font-size:26px; margin-bottom:8px; }}
.smc-qa .smc-qa-title {{ font-size:13px; font-weight:700; color:{TEXT}; }}
.smc-qa .smc-qa-sub {{ font-size:11px; color:{MUTED}; margin-top:3px; }}

/* Quick-action "Open" buttons: slim, flush under the tile above them.
   Scoped via a marker div + sibling selector so ONLY buttons in the
   Quick Actions row get this look — every other button stays the
   bold gold CTA style. */
.smc-qa-row + div [data-testid="stButton"] > button {{
    background:{PANEL_ALT} !important; color:{MUTED} !important;
    border:1px solid {BORDER} !important; border-top:none !important;
    border-radius:0 0 14px 14px !important;
    box-shadow:none !important; font-size:11.5px !important;
    font-weight:700 !important; padding:7px 10px !important;
    letter-spacing:0.04em !important; margin-top:-1px !important;
}}
.smc-qa-row + div [data-testid="stButton"] > button:hover {{
    background:{GOLD} !important; color:{INK} !important;
    transform:none !important; box-shadow:none !important;
}}

/* ── SESSION PILL — small live-status chip near the masthead ── */
.smc-session-pill {{
    display:inline-flex; align-items:center; gap:7px;
    background:{GREEN_SOFT}; color:{GREEN}; border:1px solid rgba(62,214,127,0.35);
    font-size:12px; font-weight:600; padding:5px 12px; border-radius:20px;
}}
.smc-session-dot {{
    width:6px; height:6px; border-radius:50%; background:{GREEN};
    animation:smc-blink 1.6s infinite;
}}
@keyframes smc-blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}

/* ── SIDEBAR NAV BADGE — small count pill (e.g. unread alerts) ── */
.smc-nav-badge {{
    margin-left:auto !important; font-size:10px !important; font-weight:800 !important;
    background:{RED} !important; color:#fff !important;
    padding:1px 7px !important; border-radius:10px !important;
    box-shadow:0 2px 6px rgba(255,92,92,0.4) !important;
}}

/* ── MOBILE / NARROW SCREEN ── purely visual adjustments, no
   Python/columns logic changes — Streamlit's own column blocks
   already wrap on narrow viewports; this just makes sure our custom
   HTML (masthead, tiles, pills, nav) wraps and scales cleanly too. */
@media (max-width: 768px) {{
    [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; }}
    [data-testid="stHorizontalBlock"] > div {{
        min-width:100% !important; flex:1 1 100% !important;
    }}
    .smc-masthead {{
        flex-direction:column !important; align-items:flex-start !important;
        gap:12px !important;
    }}
    .smc-masthead-title {{ font-size:30px !important; }}
    .smc-masthead-badge {{ width:48px !important; height:48px !important; font-size:22px !important; }}
    .smc-session-pill {{ font-size:11px !important; padding:4px 10px !important; }}
    [data-testid="stAppViewContainer"] [data-testid="stRadio"] label {{
        padding:9px 14px !important; font-size:12px !important;
    }}
    .smc-qa-icon {{ font-size:22px !important; }}
    [data-testid="stMetricValue"] {{ font-size:24px !important; }}
}}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='smc-topbar'></div>", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style='padding:22px 16px 18px;border-bottom:1px solid {BORDER_SOFT};'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
            <div style='width:48px;height:48px;background:linear-gradient(155deg,{GOLD},#B8860B);
                border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;
                box-shadow:0 0 0 1px rgba(255,212,0,0.4), 0 8px 22px rgba(255,212,0,0.3);'>🎓</div>
            <div>
                <div style='font-family:Fraunces,serif;font-size:22px;font-weight:800;color:{TEXT};letter-spacing:0.01em;line-height:1.1;'>SmartClass<span style='color:{GOLD};'>AI</span></div>
                <div style='font-size:9.5px;color:{FAINT};letter-spacing:0.16em;margin-top:2px;'>CLASSROOM CONSOLE · PHASE 1</div>
            </div>
        </div>
    </div>
    <div style='padding:14px 16px 6px;font-size:10px;font-weight:700;
        letter-spacing:0.16em;color:{FAINT};text-transform:uppercase;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Dashboard",
        "👥  Class Analysis",
        "📷  Single Student",
        "🏫  Classroom View",
        "📈  Analytics",
        "🤖  AI Report",
        "🔔  Alerts",
        "📁  History",
        "ℹ   About",
    ], label_visibility="collapsed", key="nav_page")

    st.markdown(f"""
    <div style='margin:16px 10px 0;background:{GOLD_SOFT};
        border:1px solid rgba(255,212,0,0.25);border-radius:10px;padding:14px;'>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
            <div style='width:7px;height:7px;background:{GREEN};border-radius:50%;
                box-shadow:0 0 0 3px rgba(62,214,127,0.25);'></div>
            <span style='font-size:12px;font-weight:700;color:{TEXT};'>System Online</span>
        </div>
        <div style='font-size:11px;color:{MUTED};line-height:1.9;font-family:JetBrains Mono,monospace;'>
            ONNX Emotion Model · 66.6%<br>
            MediaPipe Face Mesh<br>
            Gemini 1.5 Flash · Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Session controls
    st.markdown(f"""
    <div style='margin:8px 10px 0;padding:12px;'>
        <div style='font-size:10px;color:{FAINT};text-transform:uppercase;
            letter-spacing:0.14em;margin-bottom:8px;'>Session Controls</div>
    </div>
    """, unsafe_allow_html=True)

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🗑️ Clear Log"):
            st.session_state["confirm_sidebar_action"] = "clear_log"
    with col_sb2:
        if st.button("🔄 Reset"):
            st.session_state["confirm_sidebar_action"] = "reset"

    _sb_action = st.session_state.get("confirm_sidebar_action")
    if _sb_action:
        _label = "clear all session logs" if _sb_action == "clear_log" else "reset this session"
        st.markdown(f"""
        <div style='margin:6px 10px 6px;padding:10px 12px;background:{AMBER_SOFT};
            border:1px solid rgba(255,159,28,0.35);border-radius:8px;
            font-size:11.5px;color:{AMBER};line-height:1.5;'>
            ⚠️ Really {_label}? This can't be undone.
        </div>""", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✓ Yes", key="confirm_sidebar_yes"):
                if _sb_action == "clear_log":
                    from utils.logger import clear_log
                    clear_log()
                    st.session_state.class_data = []
                st.session_state.agent.clear()
                st.session_state.alerts = []
                st.session_state["confirm_sidebar_action"] = None
                st.rerun()
        with cc2:
            if st.button("✕ Cancel", key="confirm_sidebar_no"):
                st.session_state["confirm_sidebar_action"] = None
                st.rerun()

    st.markdown(f"""
    <div style='margin:12px 10px 0;padding:12px;border-top:1px solid {BORDER_SOFT};'>
        <div style='font-size:10px;color:{FAINT};text-transform:uppercase;
            letter-spacing:0.14em;margin-bottom:8px;'>Developer</div>
        <div style='display:flex;align-items:center;gap:9px;'>
            <div style='width:32px;height:32px;border-radius:50%;
                background:{GOLD}; color:{INK};
                display:flex;align-items:center;justify-content:center;
                font-size:12px;font-weight:800;'>H</div>
            <div>
                <div style='font-size:12px;font-weight:700;color:{TEXT};'>Hamna Munir</div>
                <div style='font-size:10px;color:{FAINT};'>Developer</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — SHELL
# ══════════════════════════════════════════════════════════════════
def card(content, padding="20px 22px", radius="14px", border=BORDER, extra=""):
    return f"""<div class='smc-card' style='background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%);
        border:1px solid {border};
        border-radius:{radius};padding:{padding};margin-bottom:14px;
        box-shadow:{SHADOW_SM};{extra}'>{content}</div>"""

def section_title(icon, title):
    return f"""<div style='display:flex;align-items:center;gap:9px;margin-bottom:14px;'>
        <div style='width:30px;height:30px;background:linear-gradient(155deg,{GOLD},{GOLD_DIM});
            border-radius:8px;display:flex;align-items:center;justify-content:center;
            font-size:14px;box-shadow:0 4px 12px rgba(255,212,0,0.25);'>{icon}</div>
        <div style='font-size:12.5px;font-weight:800;color:{TEXT};
            text-transform:uppercase;letter-spacing:0.09em;'>{title}</div>
    </div>"""

def engagement_badge(score):
    if score >= 70:
        return f"<span style='background:{GREEN_SOFT};color:{GREEN};border:1px solid rgba(62,214,127,0.4);padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>🟢 {score}% Focused</span>"
    elif score >= 40:
        return f"<span style='background:{AMBER_SOFT};color:{AMBER};border:1px solid rgba(255,159,28,0.4);padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>🟡 {score}% Moderate</span>"
    else:
        return f"<span style='background:{RED_SOFT};color:{RED};border:1px solid rgba(255,92,92,0.4);padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>🔴 {score}% Distracted</span>"

def state_colors(score):
    if score >= 70:   return GREEN, GREEN_SOFT, "rgba(62,214,127,0.35)"
    elif score >= 40: return AMBER, AMBER_SOFT, "rgba(255,159,28,0.35)"
    else:             return RED,   RED_SOFT,   "rgba(255,92,92,0.35)"

def student_card_html(sid, name, score, emotion, confidence, state):
    sc, bg, border = state_colors(score)
    emoji = EMOJI_MAP.get(emotion,"😐")
    initials = "".join([w[0].upper() for w in name.split()[:2]]) or "?"
    return f"""
    <div style='background:{bg};border:1px solid {border};border-radius:14px;
        padding:16px;display:flex;align-items:center;gap:14px;margin-bottom:10px;
        box-shadow:{SHADOW_XS};transition:transform 0.2s {EASE}, box-shadow 0.2s {EASE};'>
        <div style='width:44px;height:44px;border-radius:50%;background:{PANEL};
            border:2px solid {border};display:flex;align-items:center;
            justify-content:center;font-size:15px;font-weight:700;color:{sc};
            flex-shrink:0;box-shadow:0 3px 10px rgba(0,0,0,0.3);'>{initials}</div>
        <div style='flex:1;min-width:0;'>
            <div style='font-size:14px;font-weight:700;color:{TEXT};margin-bottom:3px;'>{name}</div>
            <div style='font-size:12px;color:{MUTED};'>{emoji} {emotion} · {confidence:.0f}% conf</div>
        </div>
        <div style='text-align:right;flex-shrink:0;'>
            <div class='smc-mono' style='font-size:22px;font-weight:700;color:{sc};line-height:1;'>{score}%</div>
            <div style='font-size:11px;color:{MUTED};margin-top:2px;'>{state}</div>
        </div>
        <div style='width:4px;height:44px;background:{sc};border-radius:2px;flex-shrink:0;
            box-shadow:0 0 8px {sc};'></div>
    </div>"""

def emotion_bars(all_scores):
    html = ""
    for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        color = EMOTION_COLORS.get(emo, GOLD)
        html += f"""<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>
            <div style='width:72px;font-size:12px;color:{MUTED};font-weight:600;'>{emo}</div>
            <div style='flex:1;height:6px;background:{PANEL_ALT};border-radius:3px;overflow:hidden;'>
                <div style='width:{score}%;height:100%;background:{color};border-radius:3px;'></div>
            </div>
            <div class='smc-mono' style='width:38px;font-size:12px;color:{MUTED};text-align:right;'>{score:.0f}%</div>
        </div>"""
    return html

# ── SIGNATURE ELEMENT: the analog Engagement Dial ──
def _polar(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return cx + r*math.cos(rad), cy - r*math.sin(rad)

def _arc_path(cx, cy, r, a1, a2):
    x1,y1 = _polar(cx,cy,r,a1); x2,y2 = _polar(cx,cy,r,a2)
    large = 1 if abs(a1-a2) > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f}"

def _score_angle(score):
    score = max(0, min(100, score))
    return 180 - (score/100)*180

def engagement_dial_svg(score, width=220):
    """Builds the analog engagement gauge as a base64 data-URI <img>
    instead of raw inline <svg>. Embedding a live <svg> node in the
    middle of a larger HTML string can confuse Streamlit's HTML
    renderer and cause everything *after* it in the same markdown call
    to fall back to plain escaped text (exactly the bug where the dial
    showed correctly but the card below it printed as raw HTML tags).
    A base64 image sidesteps that entirely — it's just one opaque
    <img> tag, no nested markup for the parser to trip over.
    """
    score = int(round(score))
    cx, cy, r, tw = 110, 96, 76, 15
    a40, a70 = _score_angle(40), _score_angle(70)
    needle_angle = _score_angle(score)
    nx, ny = _polar(cx, cy, r-24, needle_angle)
    sc, _, _ = state_colors(score)
    view_w, view_h = 220, 172
    height = int(width * view_h/view_w)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {view_h}" width="{view_w}" height="{view_h}">
      <path d="{_arc_path(cx,cy,r,180,a40)}" stroke="{RED}"   stroke-width="{tw}" fill="none" stroke-linecap="round"/>
      <path d="{_arc_path(cx,cy,r,a40,a70)}" stroke="{AMBER}" stroke-width="{tw}" fill="none" stroke-linecap="round"/>
      <path d="{_arc_path(cx,cy,r,a70,0)}"   stroke="{GREEN}" stroke-width="{tw}" fill="none" stroke-linecap="round"/>
      <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{TEXT}" stroke-width="4" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="8" fill="{GOLD}" stroke="{INK}" stroke-width="2"/>
      <text x="{cx}" y="{cy+46}" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="42" font-weight="800" fill="{sc}">{score}</text>
      <text x="{cx}" y="{cy+68}" text-anchor="middle" font-family="Inter, sans-serif"
            font-size="11" letter-spacing="2" fill="{MUTED}">ENGAGEMENT / 100</text>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return f"""
    <div style='display:flex;justify-content:center;margin-bottom:4px;'>
        <img src='data:image/svg+xml;base64,{b64}' width='{width}' height='{height}' alt='engagement dial' />
    </div>
    """

def render_big_score_card(score, state, emotion, confidence, blinks, ear, looking):
    """Renders the Live Readout card in TWO separate st.markdown calls
    instead of one giant combined HTML string. Each call is small,
    self-contained, and independently guaranteed to have
    unsafe_allow_html=True — this removes any possibility of one half
    of a huge combined string failing to render as HTML while the
    other half (the dial image) still shows correctly, which is
    exactly the symptom that kept showing up."""
    sc, bg, border = state_colors(score)
    emoji = EMOJI_MAP.get(emotion,"😐")
    look  = "✅ Yes" if looking else "❌ No"
    status_text = "⚠️ Alert" if score < 35 else "✅ OK"

    header_and_dial = f"""
    <div style='background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%);
        border:1px solid {border};border-radius:16px 16px 0 0;
        padding:22px 22px 0 22px;box-shadow:{SHADOW_MD};'>
        <div style='font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:0.16em;color:{sc};margin-bottom:6px;opacity:0.85;'>
            Live Readout</div>
        {engagement_dial_svg(score)}
    </div>
    """
    st.markdown(header_and_dial, unsafe_allow_html=True)

    details = f"""
    <div style='background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%);
        border:1px solid {border};border-top:none;
        border-radius:0 0 16px 16px;padding:16px 22px 22px 22px;margin-bottom:14px;
        box-shadow:0 8px 20px rgba(0,0,0,0.35);'>
        <div style='height:1px;background:{BORDER};margin-bottom:16px;'></div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
            <div style='background:{bg};border:1px solid {border};border-radius:9px;padding:10px;
                box-shadow:{SHADOW_XS};'>
                <div style='font-size:10px;color:{sc};text-transform:uppercase;
                    letter-spacing:0.1em;opacity:0.8;margin-bottom:4px;'>Emotion</div>
                <div style='font-size:14px;font-weight:700;color:{TEXT};'>{emoji} {emotion}</div>
                <div style='font-size:11px;color:{MUTED};'>{confidence:.0f}% confidence</div>
            </div>
            <div style='background:{bg};border:1px solid {border};border-radius:9px;padding:10px;
                box-shadow:{SHADOW_XS};'>
                <div style='font-size:10px;color:{sc};text-transform:uppercase;
                    letter-spacing:0.1em;opacity:0.8;margin-bottom:4px;'>Eye Contact</div>
                <div style='font-size:14px;font-weight:700;color:{TEXT};'>{look}</div>
                <div style='font-size:11px;color:{MUTED};'>EAR: {ear}</div>
            </div>
            <div style='background:{bg};border:1px solid {border};border-radius:9px;padding:10px;
                box-shadow:{SHADOW_XS};'>
                <div style='font-size:10px;color:{sc};text-transform:uppercase;
                    letter-spacing:0.1em;opacity:0.8;margin-bottom:4px;'>Blinks</div>
                <div class='smc-mono' style='font-size:18px;font-weight:700;color:{TEXT};'>{blinks}</div>
            </div>
            <div style='background:{bg};border:1px solid {border};border-radius:9px;padding:10px;
                box-shadow:{SHADOW_XS};'>
                <div style='font-size:10px;color:{sc};text-transform:uppercase;
                    letter-spacing:0.1em;opacity:0.8;margin-bottom:4px;'>Status</div>
                <div style='font-size:13px;font-weight:700;color:{sc};'>{status_text}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(details, unsafe_allow_html=True)

# Charts sit on a clean white surface (part of the black/white/yellow
# palette) with dark text and gridlines — this is what makes the data
# readable instead of muddy shaded bands on a dark background.
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    font=dict(family="Inter", color="#000000", size=13),
    title_font=dict(family="Fraunces", color="#000000", size=16),
    legend=dict(font=dict(color="#000000", size=12)),
    xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
               linecolor="#999999", showline=True,
               tickfont=dict(color="#000000", size=12),
               title=dict(font=dict(color="#000000", size=13))),
    yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
               linecolor="#999999", showline=True,
               tickfont=dict(color="#000000", size=12),
               title=dict(font=dict(color="#000000", size=13))),
    margin=dict(t=52,b=28,l=16,r=16)
)

def add_engagement_zones(fig):
    """Clear, thin threshold lines instead of translucent bands — much
    easier to read than shaded rectangles on top of the data line."""
    fig.add_hline(y=70, line_dash="dot", line_color=GREEN, line_width=1.5,
                   annotation_text="Focused ≥70", annotation_font_color=GREEN,
                   annotation_font_size=11, annotation_position="top left")
    fig.add_hline(y=40, line_dash="dot", line_color=RED, line_width=1.5,
                   annotation_text="Distracted <40", annotation_font_color=RED,
                   annotation_font_size=11, annotation_position="bottom left")
    return fig

def render_masthead():
    elapsed_min = int((time.time() - st.session_state.session_start) / 60)
    st.markdown(f"""
    <div class='smc-masthead' style='justify-content:space-between;'>
        <div style='display:flex;align-items:center;gap:20px;'>
            <div class='smc-masthead-badge'>🎓</div>
            <div>
                <div class='smc-masthead-title'>SmartClass AI</div>
                <div class='smc-masthead-tag'>Classroom Intelligence Console</div>
            </div>
        </div>
        <div class='smc-session-pill'>
            <span class='smc-session-dot'></span>
            Session active · {elapsed_min} min
        </div>
    </div>
    """, unsafe_allow_html=True)

def page_header(icon, title, subtitle=""):
    sub = f"<div style='font-size:14px;color:{MUTED};margin-top:4px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""<div style='margin-bottom:22px;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <span style='font-size:22px;'>{icon}</span>
            <div>
                <div style='font-family:Fraunces,serif;font-size:24px;font-weight:700;color:{TEXT};line-height:1.2;'>{title}</div>
                {sub}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def empty_state(icon, title, subtitle):
    st.markdown(f"""
    <div style='text-align:center;padding:56px 32px;
        background:linear-gradient(165deg, {PANEL} 0%, {PANEL_ALT} 100%);
        border:1.5px dashed {BORDER};border-radius:16px;
        box-shadow:0 2px 10px rgba(0,0,0,0.25) inset;'>
        <div style='font-size:44px;margin-bottom:12px;opacity:0.9;'>{icon}</div>
        <div style='font-size:16px;font-weight:700;color:{TEXT};margin-bottom:6px;'>{title}</div>
        <div style='font-size:13px;color:{MUTED};'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CAPTURE — unified Photo Upload / Video Upload / Live Snapshot
# ══════════════════════════════════════════════════════════════════
def capture_selector(key_prefix):
    """Renders the 3-way capture method control. Returns the chosen mode label."""
    st.markdown(card(
        section_title("🎛️","Capture Method") +
        f"<div style='font-size:12.5px;color:{MUTED};margin-bottom:12px;'>"
        "Choose how you'd like to bring in classroom footage for analysis.</div>"
    ), unsafe_allow_html=True)
    mode = st.radio(
        "Capture method", ["📸 Photo Upload", "🎥 Video Upload", "📷 Live Snapshot"],
        horizontal=True, label_visibility="collapsed", key=f"{key_prefix}_mode"
    )
    return mode

def analyze_single_frame(img_bgr, scorer):
    """Runs emotion + engagement scoring on one frame. Never raises —
    returns (emo, eng, error). If anything in the detection pipeline
    fails (corrupt frame, unreadable image, a model hiccup, etc.), emo
    and eng come back as None and error holds a short, friendly
    message, so callers can show a warning instead of the whole page
    crashing with a traceback."""
    try:
        emo = detect_emotion(img_bgr)
        eng = scorer.score(img_bgr, emo["emotion"])
        return emo, eng, None
    except Exception as e:
        return None, None, f"Couldn't analyze this image ({type(e).__name__}: {e})"

def draw_face_box(img_np, img_bgr, emo, eng):
    """Draws the face-box overlay. Wrapped defensively — if the box
    coordinates are malformed for any reason, we fall back to the
    plain (undecorated) image instead of crashing the page."""
    img_draw = img_np.copy()
    try:
        if emo and emo.get("face_box"):
            x,y,w,h = emo["face_box"]
            score = eng["engagement_score"]
            sc, _, _ = state_colors(score)
            c = tuple(int(sc.lstrip("#")[i:i+2],16) for i in (4,2,0))  # BGR from hex
            cv2.rectangle(img_draw,(x,y),(x+w,y+h),c,2)
            cv2.putText(img_draw,f"{emo['emotion']} | {score}%",
                       (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.65,c,2)
    except Exception:
        pass
    return img_draw

def analyze_video_frames(uploaded_file, name="Video Sample", max_samples=10):
    """Samples evenly-spaced frames from an uploaded video and scores
    each one. Robust to failure at every stage: if the file itself
    can't be read, returns an empty list with an error message instead
    of raising; if an individual sampled frame fails to analyze, that
    frame is skipped (logged in `skipped`) and sampling continues
    rather than aborting the whole video."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    results, skipped, error = [], 0, None

    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.close()
    except Exception as e:
        return results, f"Couldn't save the uploaded video ({e})"

    cap = None
    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            step = max(1, total_frames // max_samples)
            scorer = EngagementScorer()
            idx, sampled = 0, 0
            while sampled < max_samples and idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    emo = detect_emotion(frame)
                    eng = scorer.score(frame, emo["emotion"])
                    results.append({
                        "timestamp_sec": round(idx / fps, 1),
                        "emotion":       emo["emotion"],
                        "confidence":    emo["confidence"],
                        "score":         eng["engagement_score"],
                        "state":         eng["state"],
                        "alert":         eng["alert"],
                    })
                except Exception:
                    skipped += 1
                idx += step
                sampled += 1
        else:
            error = "This video has no readable frames — try a different file."
    except Exception as e:
        error = f"Couldn't process this video ({type(e).__name__}: {e})"
    finally:
        if cap is not None:
            cap.release()
        try:
            os.unlink(tfile.name)
        except OSError:
            pass

    if skipped and not error:
        error = f"{skipped} frame(s) couldn't be analyzed and were skipped."
    return results, error

# ══════════════════════════════════════════════════════════════════
# MASTHEAD — shown once, above every page
# ══════════════════════════════════════════════════════════════════
render_masthead()

# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    page_header("📊","Dashboard","Real-time classroom intelligence overview")

    df = read_log()

    # ── Top Stats Bar
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Analyzed",      len(df) if not df.empty else 0)
    c2.metric("Avg Focus",     f"{df['engagement_score'].mean():.0f}%" if not df.empty else "—")
    c3.metric("Alerts",        len(st.session_state.alerts))
    c4.metric("Top Emotion",   df["emotion"].mode()[0] if not df.empty else "—")
    c5.metric("Session Reads", len(df) if not df.empty else 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Navigation Cards — real, clickable, hover-lifting tiles
    st.markdown(section_title("🚀","Quick Actions"), unsafe_allow_html=True)
    st.markdown("<div class='smc-qa-row'></div>", unsafe_allow_html=True)

    quick_actions = [
        ("📸", "Photo Analysis", "Single student",   "📷  Single Student"),
        ("🏫", "Classroom View", "Multi-student",     "🏫  Classroom View"),
        ("📋", "Class Analysis", "Batch upload",      "👥  Class Analysis"),
        ("🤖", "AI Report",      "Gemini insights",   "🤖  AI Report"),
    ]
    qa_cols = st.columns(4)
    for col, (icon, title, sub, target) in zip(qa_cols, quick_actions):
        with col:
            st.markdown(f"""
            <div class='smc-qa'>
                <div class='smc-qa-icon'>{icon}</div>
                <div class='smc-qa-title'>{title}</div>
                <div class='smc-qa-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open →", key=f"qa_{title}"):
                st.session_state["nav_target"] = target
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        mode = capture_selector("dash")
        img_bgr = img_np = None
        emo = eng = None

        if mode == "📸 Photo Upload":
            f = st.file_uploader(
                "Upload classroom photo",
                type=["jpg","jpeg","png"],
                label_visibility="collapsed",
                key="dash_photo"
            )
            if f:
                img_np  = np.array(Image.open(f))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        elif mode == "🎥 Video Upload":
            vf = st.file_uploader(
                "Upload class recording",
                type=["mp4","mov","avi","mkv"],
                label_visibility="collapsed",
                key="dash_video"
            )
            if vf:
                with st.spinner("Sampling frames from video..."):
                    vresults, verr = analyze_video_frames(vf, name="Dashboard Clip")
                st.session_state.video_data = vresults
                if verr:
                    st.warning(f"⚠️ {verr}")
                if vresults:
                    vdf = pd.DataFrame(vresults)
                    avg = vdf["score"].mean()
                    st.markdown(f"""
                    <div style='background:{GOLD_SOFT};border:1px solid rgba(255,212,0,0.3);
                        border-radius:10px;padding:12px 16px;margin:12px 0;
                        font-size:13px;color:{TEXT};font-weight:600;'>
                        🎞️ {len(vresults)} frame(s) sampled · Avg: {avg:.0f}%
                    </div>""", unsafe_allow_html=True)
                    fig = px.line(vdf, x="timestamp_sec", y="score", markers=True,
                                 title="Engagement Across Video Timeline")
                    fig.update_traces(
                        line=dict(color=GOLD_DIM, width=3),
                        marker=dict(color="#000000", size=8,
                                   line=dict(color=GOLD, width=2))
                    )
                    fig.update_yaxes(range=[0,100])
                    fig = add_engagement_zones(fig)
                    fig.update_layout(**CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                    for r in vresults:
                        log_entry("Video", r["emotion"], r["confidence"],
                                 r["score"], r["state"], r["alert"])
                        if r["alert"]:
                            st.session_state.alerts.append({
                                "time":  time.strftime("%H:%M:%S"),
                                "score": r["score"],
                                "state": r["state"],
                                "msg":   f"Video @{r['timestamp_sec']}s: {r['state']} ({r['score']}%)"
                            })
                elif not verr:
                    st.warning("Couldn't read video — try a different file.")

        else:  # Live Snapshot
            snap = st.camera_input(
                "Snapshot",
                key="dash_snap",
                label_visibility="collapsed"
            )
            if snap:
                img_np  = np.array(Image.open(snap))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if img_bgr is not None:
            with st.spinner("Analyzing..."):
                if "dash_scorer" not in st.session_state:
                    st.session_state.dash_scorer = EngagementScorer()
                emo, eng, aerr = analyze_single_frame(
                    img_bgr, st.session_state.dash_scorer)

            if aerr:
                st.error(f"⚠️ {aerr}")
            else:
                img_draw = draw_face_box(img_np, img_bgr, emo, eng)
                st.image(img_draw, channels="RGB", use_column_width=True)

                log_entry("Student_Live", emo["emotion"], emo["confidence"],
                         eng["engagement_score"], eng["state"], eng["alert"])

                if eng["alert"]:
                    st.error(f"⚠️ Low focus! Score: {eng['engagement_score']}%")
                    st.session_state.alerts.append({
                        "time":  time.strftime("%H:%M:%S"),
                        "score": eng["engagement_score"],
                        "state": eng["state"],
                        "msg":   f"Live: {eng['state']} ({eng['engagement_score']}%)"
                    })
                elif eng["engagement_score"] >= 70:
                    st.success(f"✅ Highly focused! Score: {eng['engagement_score']}%")
                else:
                    st.warning(f"⚡ Moderate focus. Score: {eng['engagement_score']}%")

                if emo["all_scores"]:
                    st.markdown(card(
                        section_title("😊","Emotion Breakdown") +
                        emotion_bars(emo["all_scores"])
                    ), unsafe_allow_html=True)

    with col_r:
        if img_bgr is not None and emo is not None:
            render_big_score_card(
                eng["engagement_score"], eng["state"],
                emo["emotion"], emo["confidence"],
                eng["blinks"], eng["ear"], eng["looking_forward"]
            )

            # Score breakdown
            if "raw_scores" in eng:
                rs = eng["raw_scores"]
                breakdown_html = section_title("📊","Score Breakdown")
                for label, val in [
                    ("👁 Eye Openness", rs.get("ear_score",0)),
                    ("🎯 Head Pose",    rs.get("pose_score",0)),
                    ("🌲 Model Score",  rs.get("model_score",0)),
                ]:
                    sc, _, _ = state_colors(val)
                    breakdown_html += f"""
                    <div style='display:flex;align-items:center;gap:10px;
                        margin-bottom:10px;'>
                        <div style='font-size:12px;color:{MUTED};width:110px;
                            flex-shrink:0;'>{label}</div>
                        <div style='flex:1;height:7px;background:{PANEL_ALT};
                            border-radius:4px;overflow:hidden;'>
                            <div style='width:{val}%;height:100%;
                                background:{sc};border-radius:4px;'></div>
                        </div>
                        <div class='smc-mono' style='font-size:12px;color:{MUTED};
                            width:32px;text-align:right;'>{val}</div>
                    </div>"""
                st.markdown(card(breakdown_html), unsafe_allow_html=True)
        else:
            empty_state("📊","Ready to Analyze","Upload photo, video or take snapshot")

        # Session summary if data exists
        if not df.empty:
            summary = get_session_summary(df)
            st.markdown(card(f"""
            <div style='font-size:12px;font-weight:800;text-transform:uppercase;
                letter-spacing:0.1em;color:{GOLD};margin-bottom:12px;'>
                Session Summary</div>
            <div style='display:flex;flex-direction:column;gap:8px;'>
                <div style='display:flex;justify-content:space-between;
                    padding:8px 0;border-bottom:1px solid {BORDER_SOFT};'>
                    <span style='font-size:13px;color:{MUTED};'>Total Readings</span>
                    <span class='smc-mono' style='font-size:13px;color:{TEXT};
                        font-weight:600;'>{summary['total_readings']}</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                    padding:8px 0;border-bottom:1px solid {BORDER_SOFT};'>
                    <span style='font-size:13px;color:{MUTED};'>Avg Engagement</span>
                    <span class='smc-mono' style='font-size:13px;color:{GREEN};
                        font-weight:600;'>{summary['avg_engagement']}%</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                    padding:8px 0;border-bottom:1px solid {BORDER_SOFT};'>
                    <span style='font-size:13px;color:{MUTED};'>Dominant State</span>
                    <span style='font-size:13px;color:{TEXT};
                        font-weight:600;'>{summary['dominant_state']}</span>
                </div>
                <div style='display:flex;justify-content:space-between;
                    padding:8px 0;border-bottom:1px solid {BORDER_SOFT};'>
                    <span style='font-size:13px;color:{MUTED};'>Top Emotion</span>
                    <span style='font-size:13px;color:{TEXT};
                        font-weight:600;'>{summary['dominant_emotion']}</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:8px 0;'>
                    <span style='font-size:13px;color:{MUTED};'>Duration</span>
                    <span class='smc-mono' style='font-size:13px;color:{GOLD};
                        font-weight:600;'>{summary['duration_min']} min</span>
                </div>
            </div>
            """), unsafe_allow_html=True)

        # Recent alerts
        if st.session_state.alerts:
            alerts_html = section_title("🔔","Recent Alerts")
            for a in reversed(st.session_state.alerts[-4:]):
                alerts_html += f"""
                <div style='display:flex;justify-content:space-between;
                    align-items:center;padding:10px 12px;background:{RED_SOFT};
                    border:1px solid rgba(255,92,92,0.3);border-radius:8px;
                    margin-bottom:6px;'>
                    <div>
                        <div style='font-size:13px;font-weight:700;color:{RED};'>
                            {a['state']}</div>
                        <div style='font-size:11px;color:{MUTED};margin-top:2px;'>
                            {a['msg'][:50]}...</div>
                    </div>
                    <div class='smc-mono' style='font-size:11px;color:{MUTED};'>
                        {a['time']}</div>
                </div>"""
            st.markdown(card(alerts_html), unsafe_allow_html=True)

    # ── Analytics preview
    if not df.empty and len(df) > 3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(card(section_title("📈","Session Analytics Preview")),
                   unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = px.area(df, x="timestamp", y="engagement_score",
                         title="Engagement Timeline",
                         color_discrete_sequence=[GOLD_DIM])
            fig.update_yaxes(range=[0,100])
            fig = add_engagement_zones(fig)
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            emo_df = df["emotion"].value_counts().reset_index()
            emo_df.columns = ["emotion","count"]
            fig2 = px.pie(emo_df, names="emotion", values="count",
                         title="Emotion Distribution", hole=0.5,
                         color="emotion",
                         color_discrete_map=EMOTION_COLORS)
            fig2.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: CLASS ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "👥  Class Analysis":
    page_header("👥","Class Analysis","Batch-upload student photos for a full class engagement report")

    st.markdown(card(
        section_title("📤","Upload Student Photos") +
        f"<div style='font-size:13px;color:{MUTED};margin-bottom:4px;'>"
        "Upload individual student photos — each will be analyzed separately. "
        "Name your files as student names (e.g. Ahmed.jpg, Sara.png)</div>"
    ), unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload student photos",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"""
        <div style='background:rgba(255,212,0,0.10);border:1px solid rgba(255,212,0,0.3);border-radius:10px;
            padding:12px 16px;margin-bottom:16px;font-size:13px;color:{TEAL};font-weight:600;'>
            📊 {len(uploaded_files)} student photo(s) uploaded — analyzing...
        </div>""", unsafe_allow_html=True)

        results = []
        failed = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded_files):
            name = f.name.split(".")[0].replace("_"," ").title()
            try:
                img_pil  = Image.open(f)
                img_np   = np.array(img_pil)
                img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                emo = detect_emotion(img_bgr)
                sid = f"Student_{i+1}"
                if sid not in st.session_state.scorers:
                    st.session_state.scorers[sid] = EngagementScorer()
                eng = st.session_state.scorers[sid].score(img_bgr, emo["emotion"])

                results.append({
                    "id":         sid,
                    "name":       name,
                    "emotion":    emo["emotion"],
                    "confidence": emo["confidence"],
                    "score":      eng["engagement_score"],
                    "state":      eng["state"],
                    "alert":      eng["alert"],
                    "all_scores": emo["all_scores"],
                    "ear":        eng["ear"],
                    "blinks":     eng["blinks"],
                    "looking":    eng["looking_forward"],
                    "file":       f,
                })

                log_entry(name, emo["emotion"], emo["confidence"],
                         eng["engagement_score"], eng["state"], eng["alert"])

                if eng["alert"]:
                    st.session_state.alerts.append({
                        "time":  time.strftime("%H:%M:%S"),
                        "score": eng["engagement_score"],
                        "state": eng["state"],
                        "msg":   f"{name}: {eng['state']} ({eng['engagement_score']}%)"
                    })
            except Exception as e:
                failed.append((name, str(e)))

            progress.progress((i+1)/len(uploaded_files))

        if failed:
            names = ", ".join(n for n,_ in failed)
            st.warning(f"⚠️ {len(failed)} photo(s) couldn't be analyzed and were skipped: {names}")

        st.session_state.class_data = results

        if not results:
            st.error("None of the uploaded photos could be analyzed. Try different files.")
            st.stop()

        avg_score  = sum(r["score"] for r in results) / len(results)
        focused    = sum(1 for r in results if r["score"] >= 70)
        moderate   = sum(1 for r in results if 40 <= r["score"] < 70)
        distracted = sum(1 for r in results if r["score"] < 40)

        st.markdown(f"""
        <div style='background:{PANEL};border:1px solid {BORDER};border-radius:12px;
            padding:18px 22px;margin-bottom:16px;'>
            <div style='font-size:13px;font-weight:700;color:{TEXT};margin-bottom:14px;'>
                Class Summary — {len(results)} Students</div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;'>
                <div style='text-align:center;'>
                    <div class='smc-mono' style='font-size:28px;font-weight:700;color:{GOLD};'>{avg_score:.0f}%</div>
                    <div style='font-size:11px;color:{MUTED};'>Avg Engagement</div>
                </div>
                <div style='text-align:center;'>
                    <div class='smc-mono' style='font-size:28px;font-weight:700;color:{GREEN};'>{focused}</div>
                    <div style='font-size:11px;color:{MUTED};'>Focused</div>
                </div>
                <div style='text-align:center;'>
                    <div class='smc-mono' style='font-size:28px;font-weight:700;color:{AMBER};'>{moderate}</div>
                    <div style='font-size:11px;color:{MUTED};'>Moderate</div>
                </div>
                <div style='text-align:center;'>
                    <div class='smc-mono' style='font-size:28px;font-weight:700;color:{RED};'>{distracted}</div>
                    <div style='font-size:11px;color:{MUTED};'>Distracted</div>
                </div>
            </div>
            <div style='margin-top:14px;height:10px;background:{PANEL_ALT};border-radius:5px;overflow:hidden;display:flex;'>
                <div style='width:{focused/len(results)*100:.0f}%;background:{GREEN};'></div>
                <div style='width:{moderate/len(results)*100:.0f}%;background:{AMBER};'></div>
                <div style='width:{distracted/len(results)*100:.0f}%;background:{RED};'></div>
            </div>
            <div style='display:flex;gap:16px;margin-top:8px;'>
                <span style='font-size:11px;color:{MUTED};'>🟢 Focused</span>
                <span style='font-size:11px;color:{MUTED};'>🟡 Moderate</span>
                <span style='font-size:11px;color:{MUTED};'>🔴 Distracted</span>
            </div>
        </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Student List", "🖼️ Photo Grid", "⚠️ Needs Attention"])

        with tab1:
            sorted_results = sorted(results, key=lambda x: x["score"])
            for r in sorted_results:
                st.markdown(student_card_html(
                    r["id"], r["name"], r["score"],
                    r["emotion"], r["confidence"], r["state"]
                ), unsafe_allow_html=True)

        with tab2:
            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    img = Image.open(r["file"])
                    st.image(img, caption=f"{r['name']} · {r['score']}%", use_column_width=True)
                    st.markdown(engagement_badge(r["score"]), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        with tab3:
            alert_students = [r for r in results if r["alert"]]
            if not alert_students:
                st.success("✅ No distracted students detected in this batch!")
            else:
                for r in alert_students:
                    st.markdown(f"""
                    <div style='background:{RED_SOFT};border:1px solid rgba(255,92,92,0.35);border-radius:10px;
                        padding:14px 16px;margin-bottom:10px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;'>
                            <div>
                                <div style='font-size:14px;font-weight:700;color:{RED};'>
                                    ⚠️ {r["name"]}</div>
                                <div style='font-size:12px;color:{MUTED};margin-top:3px;'>
                                    {r["emotion"]} · Score: {r["score"]}% · {r["state"]}</div>
                            </div>
                            <div class='smc-mono' style='font-size:22px;font-weight:700;color:{RED};'>{r["score"]}%</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

    else:
        empty_state("👥","Upload student photos to begin","Supports multiple files — each analyzed individually")

# ══════════════════════════════════════════════════════════════════
# PAGE: SINGLE STUDENT
# ══════════════════════════════════════════════════════════════════
elif page == "📷  Single Student":
    page_header("📷","Single Student","Deep-dive focus & emotion analysis")

    col_left, col_right = st.columns([1,1])

    with col_left:
        mode = st.radio(
            "Input", ["📸 Photo Upload","🎥 Video Upload","📷 Live Snapshot"],
            horizontal=True, key="ss_mode", label_visibility="collapsed"
        )

        student_name = st.text_input(
            "Student name (optional)",
            placeholder="e.g. Ahmed Khan",
            key="ss_name"
        )

        img_bgr = img_np = None
        emo = eng = None

        if mode == "📸 Photo Upload":
            f = st.file_uploader(
                "Upload photo",
                type=["jpg","jpeg","png"],
                label_visibility="collapsed",
                key="ss_photo"
            )
            if f:
                img_np  = np.array(Image.open(f))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        elif mode == "🎥 Video Upload":
            vf = st.file_uploader(
                "Upload video",
                type=["mp4","mov","avi","mkv"],
                label_visibility="collapsed",
                key="ss_video"
            )
            if vf:
                name = student_name or "Student"
                with st.spinner(f"Analyzing {name}'s video..."):
                    vresults, verr = analyze_video_frames(vf, name=name, max_samples=15)
                if verr:
                    st.warning(f"⚠️ {verr}")
                if vresults:
                    vdf = pd.DataFrame(vresults)
                    avg = vdf["score"].mean()
                    foc = len(vdf[vdf["score"]>=60])
                    dis = len(vdf[vdf["score"]<35])

                    st.markdown(f"""
                    <div style='background:{GOLD_SOFT};border:1px solid rgba(255,212,0,0.3);
                        border-radius:10px;padding:14px 16px;margin:12px 0;'>
                        <div style='font-size:14px;font-weight:700;color:{TEXT};
                            margin-bottom:8px;'>{name} — Video Analysis</div>
                        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                            gap:8px;'>
                            <div style='text-align:center;'>
                                <div class='smc-mono' style='font-size:22px;
                                    font-weight:700;color:{GOLD};'>{avg:.0f}%</div>
                                <div style='font-size:11px;color:{MUTED};'>Avg Focus</div>
                            </div>
                            <div style='text-align:center;'>
                                <div class='smc-mono' style='font-size:22px;
                                    font-weight:700;color:{GREEN};'>{foc}</div>
                                <div style='font-size:11px;color:{MUTED};'>Focused frames</div>
                            </div>
                            <div style='text-align:center;'>
                                <div class='smc-mono' style='font-size:22px;
                                    font-weight:700;color:{RED};'>{dis}</div>
                                <div style='font-size:11px;color:{MUTED};'>Distracted</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    fig = px.line(vdf, x="timestamp_sec", y="score",
                                 markers=True, title=f"{name} — Focus Timeline")
                    fig.update_traces(
                        line=dict(color=GOLD_DIM, width=3),
                        marker=dict(color="#000000", size=8,
                                   line=dict(color=GOLD, width=2))
                    )
                    fig.update_yaxes(range=[0,100])
                    fig = add_engagement_zones(fig)
                    fig.update_layout(**CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                    for r in vresults:
                        log_entry(name, r["emotion"], r["confidence"],
                                 r["score"], r["state"], r["alert"])
                elif not verr:
                    st.warning("Couldn't read this video — try a different file.")

        else:
            snap = st.camera_input(
                "Take snapshot",
                key="ss_snap",
                label_visibility="collapsed"
            )
            if snap:
                img_np  = np.array(Image.open(snap))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if img_bgr is not None:
            name = student_name or "Student"
            with st.spinner(f"Analyzing {name}..."):
                if "ss_scorer" not in st.session_state:
                    st.session_state.ss_scorer = EngagementScorer()
                emo, eng, aerr = analyze_single_frame(
                    img_bgr, st.session_state.ss_scorer)

            if aerr:
                st.error(f"⚠️ {aerr}")
            else:
                img_draw = draw_face_box(img_np, img_bgr, emo, eng)
                st.markdown(f"**Analyzing: {name}**")
                st.image(img_draw, channels="RGB", use_column_width=True)
                log_entry(name, emo["emotion"], emo["confidence"],
                         eng["engagement_score"], eng["state"], eng["alert"])

    with col_right:
        if img_bgr is not None and emo is not None:
            render_big_score_card(
                eng["engagement_score"], eng["state"],
                emo["emotion"], emo["confidence"],
                eng["blinks"], eng["ear"], eng["looking_forward"]
            )

            if emo["all_scores"]:
                st.markdown(card(
                    section_title("😊","Emotion Breakdown") +
                    emotion_bars(emo["all_scores"])
                ), unsafe_allow_html=True)

            # Score breakdown
            if "raw_scores" in eng:
                rs = eng["raw_scores"]
                bd_html = section_title("📊","Score Breakdown")
                for label, val in [
                    ("👁 Eye Openness", rs.get("ear_score",0)),
                    ("🎯 Head Pose",    rs.get("pose_score",0)),
                    ("🌲 Model Score",  rs.get("model_score",0)),
                ]:
                    sc, _, _ = state_colors(val)
                    bd_html += f"""
                    <div style='display:flex;align-items:center;gap:10px;
                        margin-bottom:10px;'>
                        <div style='font-size:12px;color:{MUTED};
                            width:110px;flex-shrink:0;'>{label}</div>
                        <div style='flex:1;height:7px;background:{PANEL_ALT};
                            border-radius:4px;overflow:hidden;'>
                            <div style='width:{val}%;height:100%;
                                background:{sc};border-radius:4px;'></div>
                        </div>
                        <div class='smc-mono' style='font-size:12px;
                            color:{MUTED};width:32px;text-align:right;'>{val}</div>
                    </div>"""
                st.markdown(card(bd_html), unsafe_allow_html=True)

            if eng["alert"]:
                st.error(f"⚠️ Low focus! Score: {eng['engagement_score']}%")
            elif eng["engagement_score"] >= 70:
                st.success(f"✅ Highly focused! {eng['engagement_score']}%")
            else:
                st.warning(f"⚡ Moderate focus. {eng['engagement_score']}%")
        else:
            empty_state(
                "📷",
                "Ready to analyze",
                "Upload photo, video or take a snapshot"
            )

# ══════════════════════════════════════════════════════════════════
# PAGE: CLASSROOM VIEW
# ══════════════════════════════════════════════════════════════════
elif page == "🏫  Classroom View":
    page_header("🏫","Classroom View","Multi-student real-time engagement detection")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        mode = st.radio("Input", ["📁 Upload Photo","📷 Live Snapshot"],
                       horizontal=True, key="cv_mode")
    with col_ctrl2:
        show_grid = st.checkbox("Show student grid", value=True)
    with col_ctrl3:
        if st.button("🔄 Reset Agent"):
            st.session_state["confirm_reset_agent"] = True

    if st.session_state.get("confirm_reset_agent"):
        st.warning("⚠️ Reset the agent? This clears all flagged/chronic tracking for this session.")
        rc1, rc2, _ = st.columns([1,1,4])
        with rc1:
            if st.button("✓ Yes, reset", key="confirm_reset_agent_yes"):
                st.session_state.agent.clear()
                st.session_state["confirm_reset_agent"] = False
                st.success("Agent reset!")
        with rc2:
            if st.button("✕ Cancel", key="confirm_reset_agent_no"):
                st.session_state["confirm_reset_agent"] = False
                st.rerun()

    # Input
    img_file = None
    if "Upload" in mode:
        img_file = st.file_uploader(
            "Upload classroom photo",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed",
            key="cv_photo"
        )
    else:
        img_file = st.camera_input(
            "Classroom snapshot",
            key="cv_snap",
            label_visibility="collapsed"
        )

    if img_file:
        img_pil = Image.open(img_file)
        img_np  = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        try:
            with st.spinner("Detecting all students..."):
                students, annotated = analyze_classroom(img_bgr)
                summary             = get_class_summary(students)

            # Run agent
            new_alerts = st.session_state.agent.update(students)
        except Exception as e:
            st.error(f"⚠️ Couldn't analyze this classroom photo "
                     f"({type(e).__name__}: {e}). Try a clearer or different image.")
            st.stop()

        # Add to session alerts
        for a in new_alerts:
            st.session_state.alerts.append({
                "time":  a["time"],
                "score": a["score"],
                "state": a["type"],
                "msg":   a["msg"]
            })

        # ── Layout
        col_img, col_info = st.columns([3, 2])

        with col_img:
            st.markdown(card(section_title("🏫","Analyzed Classroom")),
                       unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True,
                    caption=f"{summary['total']} student(s) detected")

            # New alerts
            for a in new_alerts:
                if a["severity"] == "critical":
                    st.error(f"🚨 {a['msg']}")
                elif a["severity"] == "high":
                    st.warning(f"⚠️ {a['msg']}")
                else:
                    st.info(f"ℹ️ {a['msg']}")

        with col_info:
            if summary["total"] == 0:
                st.warning("No faces detected — try a clearer photo.")
            else:
                foc_pct  = summary["focus_pct"]
                mod_pct  = round(summary["moderate"]/summary["total"]*100)
                dist_pct = round(summary["distracted"]/summary["total"]*100)

                # Class overview card
                st.markdown(card(f"""
                <div style='font-size:12px;font-weight:800;text-transform:uppercase;
                    letter-spacing:0.1em;color:{GOLD};margin-bottom:14px;'>
                    Class Overview</div>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                    gap:10px;margin-bottom:14px;'>
                    <div style='text-align:center;background:{PANEL_ALT};
                        border-radius:10px;padding:12px;'>
                        <div class='smc-mono' style='font-size:26px;font-weight:700;
                            color:{GOLD};'>{summary['total']}</div>
                        <div style='font-size:11px;color:{MUTED};'>Students</div>
                    </div>
                    <div style='text-align:center;background:{PANEL_ALT};
                        border-radius:10px;padding:12px;'>
                        <div class='smc-mono' style='font-size:26px;font-weight:700;
                            color:{GREEN};'>{summary['avg_score']}%</div>
                        <div style='font-size:11px;color:{MUTED};'>Avg Score</div>
                    </div>
                    <div style='text-align:center;background:{PANEL_ALT};
                        border-radius:10px;padding:12px;'>
                        <div class='smc-mono' style='font-size:26px;font-weight:700;
                            color:{RED};'>{summary['alert_count']}</div>
                        <div style='font-size:11px;color:{MUTED};'>Alerts</div>
                    </div>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                    gap:8px;margin-bottom:14px;'>
                    <div style='text-align:center;background:{GREEN_SOFT};
                        border-radius:10px;padding:10px;'>
                        <div class='smc-mono' style='font-size:22px;font-weight:700;
                            color:{GREEN};'>{summary['focused']}</div>
                        <div style='font-size:11px;color:{MUTED};'>🟢 Focused</div>
                    </div>
                    <div style='text-align:center;background:{AMBER_SOFT};
                        border-radius:10px;padding:10px;'>
                        <div class='smc-mono' style='font-size:22px;font-weight:700;
                            color:{AMBER};'>{summary['moderate']}</div>
                        <div style='font-size:11px;color:{MUTED};'>🟡 Moderate</div>
                    </div>
                    <div style='text-align:center;background:{RED_SOFT};
                        border-radius:10px;padding:10px;'>
                        <div class='smc-mono' style='font-size:22px;font-weight:700;
                            color:{RED};'>{summary['distracted']}</div>
                        <div style='font-size:11px;color:{MUTED};'>🔴 Distracted</div>
                    </div>
                </div>
                <div style='height:10px;background:{PANEL_ALT};border-radius:5px;
                    overflow:hidden;display:flex;margin-bottom:8px;'>
                    <div style='width:{foc_pct}%;background:{GREEN};'></div>
                    <div style='width:{mod_pct}%;background:{AMBER};'></div>
                    <div style='width:{dist_pct}%;background:{RED};'></div>
                </div>
                <div style='display:flex;gap:12px;'>
                    <span style='font-size:11px;color:{MUTED};'>🟢 {foc_pct}%</span>
                    <span style='font-size:11px;color:{MUTED};'>🟡 {mod_pct}%</span>
                    <span style='font-size:11px;color:{MUTED};'>🔴 {dist_pct}%</span>
                </div>
                """), unsafe_allow_html=True)

                # Agent summary
                agent_sum = st.session_state.agent.get_summary()
                st.markdown(card(f"""
                <div style='font-size:12px;font-weight:800;text-transform:uppercase;
                    letter-spacing:0.1em;color:{GOLD};margin-bottom:12px;'>
                    Agent Status</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                    <div style='background:{PANEL_ALT};border-radius:8px;padding:10px;
                        text-align:center;'>
                        <div class='smc-mono' style='font-size:20px;font-weight:700;
                            color:{RED};'>{agent_sum['total_alerts']}</div>
                        <div style='font-size:11px;color:{MUTED};'>Total Alerts</div>
                    </div>
                    <div style='background:{PANEL_ALT};border-radius:8px;padding:10px;
                        text-align:center;'>
                        <div class='smc-mono' style='font-size:20px;font-weight:700;
                            color:{AMBER};'>{agent_sum['flagged_count']}</div>
                        <div style='font-size:11px;color:{MUTED};'>Flagged</div>
                    </div>
                    <div style='background:{PANEL_ALT};border-radius:8px;padding:10px;
                        text-align:center;'>
                        <div class='smc-mono' style='font-size:20px;font-weight:700;
                            color:{RED};'>{agent_sum['chronic_count']}</div>
                        <div style='font-size:11px;color:{MUTED};'>Chronic</div>
                    </div>
                    <div style='background:{PANEL_ALT};border-radius:8px;padding:10px;
                        text-align:center;'>
                        <div class='smc-mono' style='font-size:20px;font-weight:700;
                            color:{GREEN};'>{agent_sum['session_min']} min</div>
                        <div style='font-size:11px;color:{MUTED};'>Session</div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                # Top alerts
                if agent_sum["alerts"]:
                    alerts_html = f"<div style='font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;color:{GOLD};margin-bottom:12px;'>Recent Agent Alerts</div>"
                    for a in reversed(agent_sum["alerts"][-5:]):
                        sev_color = RED if a["severity"]=="critical" else AMBER if a["severity"]=="high" else GOLD
                        alerts_html += f"""
                        <div style='border-left:3px solid {sev_color};padding:8px 12px;
                            margin-bottom:6px;background:{PANEL_ALT};border-radius:0 8px 8px 0;'>
                            <div style='font-size:12px;font-weight:600;color:{sev_color};'>
                                {a['type'].replace('_',' ')}</div>
                            <div style='font-size:11px;color:{MUTED};margin-top:2px;'>{a['msg']}</div>
                            <div class='smc-mono' style='font-size:10px;color:{FAINT};margin-top:2px;'>{a['time']}</div>
                        </div>"""
                    st.markdown(card(alerts_html), unsafe_allow_html=True)

        # ── Per-student grid
        if show_grid and students:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(card(section_title("👥","Per-Student Engagement Grid")),
                       unsafe_allow_html=True)

            # Sort by score — distracted first
            sorted_students = sorted(students, key=lambda x: x["engagement"])

            cols_per_row = 5
            rows = [sorted_students[i:i+cols_per_row]
                   for i in range(0, len(sorted_students), cols_per_row)]

            for row in rows:
                cols = st.columns(cols_per_row)
                for col, s in zip(cols, row):
                    sc, bg, border = state_colors(s["engagement"])
                    look   = "✅" if s["looking_forward"] else "❌"
                    status = st.session_state.agent.get_student_status(s["id"])
                    flags  = st.session_state.agent.flagged_students.get(s["id"], 0)
                    flag_badge = f"<div style='background:{RED_SOFT};color:{RED};font-size:9px;border-radius:4px;padding:1px 5px;margin-top:3px;'>⚠️ {flags}x flagged</div>" if flags > 0 else ""

                    with col:
                        st.markdown(f"""
                        <div style='background:{PANEL};border:1px solid {border};
                            border-top:3px solid {sc};
                            border-radius:10px;padding:12px;text-align:center;
                            margin-bottom:8px;'>
                            <div class='smc-mono' style='font-size:16px;font-weight:700;
                                color:{GOLD};margin-bottom:4px;'>{s["id"]}</div>
                            <div class='smc-mono' style='font-size:28px;font-weight:800;
                                color:{sc};line-height:1;'>{s["engagement"]}%</div>
                            <div style='font-size:10px;color:{MUTED};margin-top:4px;'>
                                {s["state"]}</div>
                            <div style='font-size:11px;margin-top:4px;'>{look}</div>
                            <div style='font-size:10px;color:{MUTED};margin-top:2px;'>
                                EAR:{s["ear"]}</div>
                            {flag_badge}
                        </div>""", unsafe_allow_html=True)

            # Log all students
            for s in students:
                log_entry(s["id"], s["emotion"], 0,
                         s["engagement"], s["state"], s["alert"])

    else:
        empty_state("🏫","Upload classroom photo",
                   "Photo mein multiple students visible honay chahiye")

# ══════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    page_header("📈","Analytics","Engagement trends, emotions & session data")

    df = read_log()
    if df.empty:
        empty_state("📈","No data yet","Analyze students first!")
    else:
        # Top metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Records",   len(df))
        c2.metric("Avg Engagement",  f"{df['engagement_score'].mean():.1f}%")
        c3.metric("Peak Score",      f"{df['engagement_score'].max()}%")
        c4.metric("Total Alerts",    int(df["alert"].sum()))

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Engagement","😊 Emotions",
            "🗃️ States","👥 Per Student"
        ])

        with tab1:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = px.area(df, x="timestamp", y="engagement_score",
                         title="Engagement Score Over Time",
                         color_discrete_sequence=[GOLD_DIM])
            fig.update_traces(
                line=dict(color=GOLD, width=2),
                fillcolor=GOLD_SOFT
            )
            fig.update_yaxes(range=[0,100])
            fig = add_engagement_zones(fig)
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # Engagement distribution histogram
            fig_hist = px.histogram(
                df, x="engagement_score",
                title="Engagement Score Distribution",
                nbins=20,
                color_discrete_sequence=[GOLD_DIM]
            )
            fig_hist.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                emo_df = df["emotion"].value_counts().reset_index()
                emo_df.columns = ["emotion","count"]
                fig2 = px.pie(
                    emo_df, names="emotion", values="count",
                    title="Emotion Distribution", hole=0.5,
                    color="emotion",
                    color_discrete_map=EMOTION_COLORS
                )
                fig2.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                fig3 = px.bar(
                    emo_df, x="emotion", y="count",
                    title="Emotion Frequency",
                    color="emotion",
                    color_discrete_map=EMOTION_COLORS
                )
                fig3.update_layout(**CHART_LAYOUT, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

            # Emotion over time
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig_emo_time = px.scatter(
                df, x="timestamp", y="emotion",
                color="emotion", title="Emotion Timeline",
                color_discrete_map=EMOTION_COLORS
            )
            fig_emo_time.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_emo_time, use_container_width=True)

        with tab3:
            sdf = df["state"].value_counts().reset_index()
            sdf.columns = ["state","count"]
            fig4 = px.bar(
                sdf, x="state", y="count",
                title="Engagement States Frequency",
                color="state",
                color_discrete_sequence=[GREEN,GOLD,AMBER,RED]
            )
            fig4.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

            # Alert timeline
            alert_df = df[df["alert"]==True].copy()
            if not alert_df.empty:
                alert_df["timestamp"] = pd.to_datetime(alert_df["timestamp"])
                fig5 = px.scatter(
                    alert_df, x="timestamp",
                    y="engagement_score",
                    title="Alert Events Timeline",
                    color_discrete_sequence=[RED]
                )
                fig5.update_traces(marker=dict(size=10, symbol="x"))
                fig5.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig5, use_container_width=True)

        with tab4:
            if "student_id" in df.columns and df["student_id"].nunique() > 1:
                # Per student avg
                student_avg = df.groupby("student_id")["engagement_score"].mean().reset_index()
                student_avg.columns = ["Student","Avg Score"]
                student_avg = student_avg.sort_values("Avg Score")

                fig6 = px.bar(
                    student_avg, x="Student", y="Avg Score",
                    title="Average Engagement Per Student",
                    color="Avg Score",
                    color_continuous_scale=[[0,RED],[0.4,AMBER],[0.7,GREEN],[1,GREEN]]
                )
                fig6.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig6, use_container_width=True)

                # Box plot
                fig7 = px.box(
                    df, x="student_id", y="engagement_score",
                    title="Engagement Range Per Student",
                    color_discrete_sequence=[GOLD_DIM]
                )
                fig7.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("Analyze multiple named students to see per-student comparison.")

# ══════════════════════════════════════════════════════════════════
# PAGE: AI REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  AI Report":
    page_header("🤖","AI Session Report","Google Gemini 1.5 Flash — 3 Report Types")

    from modules.report_generator import (
        generate_class_summary,
        generate_improvement_report,
        generate_effectiveness_report
    )

    def _gemini_key_configured():
        """Best-effort check for a Gemini API key in secrets.toml. Checks
        the common key names/shapes; if secrets.toml doesn't exist at all,
        st.secrets access itself can raise, which we treat as 'not
        configured' rather than crashing."""
        try:
            for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key"):
                if k in st.secrets and st.secrets[k]:
                    return True
            if "gemini" in st.secrets and dict(st.secrets["gemini"]):
                return True
            return False
        except Exception:
            return False

    if not _gemini_key_configured():
        st.markdown(f"""
        <div style='background:{AMBER_SOFT};border:1px solid rgba(255,159,28,0.35);
            border-radius:10px;padding:12px 16px;margin-bottom:16px;
            font-size:13px;color:{AMBER};line-height:1.6;'>
            ⚠️ No Gemini API key detected in <code>.streamlit/secrets.toml</code>.
            Report generation will fail until one is added, e.g.
            <code>GEMINI_API_KEY = "your-key-here"</code>. (If your key uses a
            different name than the app expects, this warning may be a
            false alarm — safe to ignore in that case.)
        </div>""", unsafe_allow_html=True)

    df = read_log()
    if df.empty:
        empty_state("🤖","No session data yet","Analyze students first — then generate AI reports")
    else:
        summary      = get_session_summary(df)
        class_data   = st.session_state.class_data
        agent_sum    = st.session_state.agent.get_summary() if hasattr(st.session_state.agent, 'get_summary') else {}

        # ── Session metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Engagement",  f"{summary['avg_engagement']}%")
        c2.metric("Dominant State",  summary['dominant_state'])
        c3.metric("Duration",        f"{summary['duration_min']} min")
        c4.metric("Total Readings",  summary['total_readings'])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── At-risk students
        at_risk = [r for r in class_data if r.get("score",0) < 40]
        if at_risk:
            risk_html = section_title("⚠️","Students Needing Attention")
            for r in at_risk:
                risk_html += student_card_html(
                    r["id"], r["name"], r["score"],
                    r["emotion"], r["confidence"], r["state"]
                )
            st.markdown(card(risk_html), unsafe_allow_html=True)

        # ── Report type selector
        st.markdown(card(
            section_title("🤖","Select Report Type") +
            f"<div style='font-size:13px;color:{MUTED};margin-bottom:4px;'>"
            "Choose the type of AI report to generate. Each uses different data.</div>"
        ), unsafe_allow_html=True)

        report_type = st.radio("Report Type", [
            "📋 Today's Class Summary",
            "👤 Students Who Need Improvement",
            "📊 Teaching Effectiveness Report",
        ], horizontal=True, label_visibility="collapsed", key="ai_report_type")

        # Report descriptions
        descriptions = {
            "📋 Today's Class Summary":
                f"<div style='background:{PANEL_ALT};border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:13px;color:{MUTED};line-height:1.7;'>📋 Analyzes overall class engagement, emotional climate, and gives teacher a session overview with key recommendations.</div>",
            "👤 Students Who Need Improvement":
                f"<div style='background:{PANEL_ALT};border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:13px;color:{MUTED};line-height:1.7;'>👤 Identifies specific students with low engagement, describes distraction patterns, and suggests targeted intervention strategies.</div>",
            "📊 Teaching Effectiveness Report":
                f"<div style='background:{PANEL_ALT};border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:13px;color:{MUTED};line-height:1.7;'>📊 Rates teaching session effectiveness out of 10, identifies what worked, and provides one actionable improvement strategy.</div>",
        }
        st.markdown(descriptions.get(report_type,""), unsafe_allow_html=True)

        if st.button("✦ Generate AI Report"):
            report = None
            try:
                with st.spinner("Gemini is analyzing your session..."):
                    if report_type == "📋 Today's Class Summary":
                        report = generate_class_summary(summary, class_data)
                    elif report_type == "👤 Students Who Need Improvement":
                        report = generate_improvement_report(summary, class_data, agent_sum)
                    else:
                        report = generate_effectiveness_report(summary, agent_sum)
            except Exception as e:
                st.error(
                    f"⚠️ Couldn't generate the AI report ({type(e).__name__}: {e}). "
                    "This is usually a missing/invalid Gemini API key in `.streamlit/secrets.toml`, "
                    "a network issue, or a quota limit — not a bug in this page."
                )

            if report:
                # Display report
                type_icon = "📋" if "Summary" in report_type else "👤" if "Improvement" in report_type else "📊"
                st.markdown(f"""
                <div style='background:{PANEL};border:1px solid {BORDER};
                    border-left:4px solid {GOLD};border-radius:14px;
                    padding:28px 32px;margin-top:16px;
                    box-shadow:0 4px 20px rgba(255,212,0,0.08);'>
                    <div style='display:flex;align-items:center;gap:10px;margin-bottom:18px;'>
                        <div style='width:32px;height:32px;background:{GOLD_SOFT};
                            border:1px solid rgba(255,212,0,0.3);border-radius:8px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:14px;'>{type_icon}</div>
                        <div>
                            <div style='font-size:12px;font-weight:800;text-transform:uppercase;
                                letter-spacing:0.12em;color:{GOLD};'>{report_type}</div>
                            <div style='font-size:11px;color:{MUTED};margin-top:2px;'>
                                Generated by Google Gemini 1.5 Flash</div>
                        </div>
                    </div>
                    <div style='font-size:15px;color:{TEXT};line-height:2;
                        border-top:1px solid {BORDER_SOFT};padding-top:16px;'>{report}</div>
                </div>""", unsafe_allow_html=True)

        # ── Agent summary
        if agent_sum and agent_sum.get("total_alerts",0) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            ag_html = section_title("🤖","Agent Session Summary")
            ag_html += f"""
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;'>
                <div style='background:{PANEL_ALT};border-radius:8px;padding:12px;text-align:center;'>
                    <div class='smc-mono' style='font-size:22px;font-weight:700;color:{RED};'>{agent_sum['total_alerts']}</div>
                    <div style='font-size:11px;color:{MUTED};'>Alerts</div>
                </div>
                <div style='background:{PANEL_ALT};border-radius:8px;padding:12px;text-align:center;'>
                    <div class='smc-mono' style='font-size:22px;font-weight:700;color:{AMBER};'>{agent_sum['flagged_count']}</div>
                    <div style='font-size:11px;color:{MUTED};'>Flagged</div>
                </div>
                <div style='background:{PANEL_ALT};border-radius:8px;padding:12px;text-align:center;'>
                    <div class='smc-mono' style='font-size:22px;font-weight:700;color:{GREEN};'>{agent_sum['class_avg']}%</div>
                    <div style='font-size:11px;color:{MUTED};'>Class Avg</div>
                </div>
                <div style='background:{PANEL_ALT};border-radius:8px;padding:12px;text-align:center;'>
                    <div class='smc-mono' style='font-size:22px;font-weight:700;color:{GOLD};'>{agent_sum['session_min']}m</div>
                    <div style='font-size:11px;color:{MUTED};'>Duration</div>
                </div>
            </div>"""

            if agent_sum.get("alerts"):
                for a in agent_sum["alerts"][-5:]:
                    sev_color = RED if a["severity"]=="critical" else AMBER
                    ag_html += f"""
                    <div style='border-left:3px solid {sev_color};padding:8px 12px;
                        margin-bottom:6px;background:{PANEL_ALT};border-radius:0 8px 8px 0;'>
                        <div style='font-size:12px;font-weight:600;color:{sev_color};'>
                            {a['type'].replace('_',' ')}</div>
                        <div style='font-size:11px;color:{MUTED};margin-top:2px;'>{a['msg']}</div>
                        <div class='smc-mono' style='font-size:10px;color:{FAINT};margin-top:2px;'>{a['time']}</div>
                    </div>"""

            st.markdown(card(ag_html), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ══════════════════════════════════════════════════════════════════
elif page == "🔔  Alerts":
    page_header("🔔","Agent Alerts","Automated distraction & engagement alerts")

    if not st.session_state.alerts:
        empty_state("✅","No alerts yet","Alerts fire when engagement drops below 35%")
    else:
        st.markdown(f"""
        <div style='background:{RED_SOFT};border:1px solid rgba(255,92,92,0.35);border-radius:10px;
            padding:12px 16px;margin-bottom:16px;font-size:14px;color:{RED};font-weight:700;'>
            ⚠️ {len(st.session_state.alerts)} alert(s) triggered this session
        </div>""", unsafe_allow_html=True)

        for a in reversed(st.session_state.alerts):
            st.markdown(f"""
            <div style='background:{PANEL};border:1px solid {BORDER};border-left:4px solid {RED};
                border-radius:10px;padding:14px 16px;margin-bottom:10px;
                display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-size:14px;font-weight:700;color:{RED};'>⚠️ {a["state"]}</div>
                    <div style='font-size:12px;color:{MUTED};margin-top:4px;'>{a["msg"]}</div>
                </div>
                <div style='text-align:right;'>
                    <div class='smc-mono' style='font-size:11px;color:{MUTED};'>{a["time"]}</div>
                    <div class='smc-mono' style='font-size:20px;font-weight:700;color:{RED};'>{a["score"]}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear All Alerts"):
            st.session_state["confirm_clear_alerts"] = True

        if st.session_state.get("confirm_clear_alerts"):
            st.warning("⚠️ Clear all alerts in this session? This can't be undone.")
            ac1, ac2, _ = st.columns([1,1,4])
            with ac1:
                if st.button("✓ Yes, clear", key="confirm_clear_alerts_yes"):
                    st.session_state.alerts = []
                    st.session_state["confirm_clear_alerts"] = False
                    st.rerun()
            with ac2:
                if st.button("✕ Cancel", key="confirm_clear_alerts_no"):
                    st.session_state["confirm_clear_alerts"] = False
                    st.rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════
elif page == "📁  History":
    page_header("📁","Session History","All recorded engagement data")

    df = read_log()
    if df.empty:
        st.info("No history yet. Analyze some students first!")
    else:
        sorted_df = df.sort_values("timestamp", ascending=False)

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.markdown(f"""
            <div style='font-size:12px;color:{MUTED};padding-top:8px;'>
                {len(sorted_df)} total record(s) logged this session — nothing
                is ever deleted here, this only limits what's shown below.
            </div>""", unsafe_allow_html=True)
        with c2:
            show_n = st.selectbox(
                "Show", ["Last 100", "Last 500", "All"],
                index=0, label_visibility="collapsed"
            )
        with c3:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "smartclassroom_history.csv",
                "text/csv"
            )

        display_df = sorted_df
        if show_n == "Last 100":
            display_df = sorted_df.head(100)
        elif show_n == "Last 500":
            display_df = sorted_df.head(500)

        st.dataframe(
            display_df,
            use_container_width=True, height=500
        )
# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ   About":
    page_header("ℹ","About SmartClass AI","AI-powered classroom intelligence platform")

    # Hero card
    st.markdown(f"""
    <div style='background:linear-gradient(145deg,#111111,#1B1B1B);
        border:1px solid rgba(255,212,0,0.25);border-radius:20px;
        padding:48px;text-align:center;margin-bottom:24px;
        box-shadow:0 0 40px rgba(255,212,0,0.08);'>
        <div style='font-family:Fraunces,serif;font-size:48px;font-weight:900;
            background:linear-gradient(90deg,#FFFFFF,{GOLD});
            -webkit-background-clip:text;background-clip:text;
            -webkit-text-fill-color:transparent;margin-bottom:10px;'>
            SmartClass AI</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:11px;
            color:{GOLD};letter-spacing:0.25em;text-transform:uppercase;
            margin-bottom:20px;'>Classroom Intelligence Console · Phase 1</div>
        <div style='font-size:15px;color:{MUTED};max-width:560px;
            margin:0 auto;line-height:1.8;'>
            A complete AI system that monitors student attention, detects emotions,
            scores engagement in real-time, and generates intelligent teaching reports
            powered by Google Gemini.</div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Developer card — tags are built as real Python (a loop that runs
        # BEFORE the f-string), then dropped into the template as one
        # finished variable. Pasting a "for" loop as literal text inside
        # an f-string doesn't execute it — it just leaves a stray {t}
        # expression with no such variable, which is what threw the
        # NameError before.
        tags_html = "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
        for t in ["AI/ML","Computer Vision","GenAI","Deep Learning","Streamlit"]:
            tags_html += f"<span style='background:{GOLD_SOFT};color:{GOLD};border:1px solid rgba(255,212,0,0.3);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;'>{t}</span>"
        tags_html += "</div>"

        st.markdown(card(f"""
        {section_title("👩‍💻","Developer")}
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:18px;'>
            <div style='width:56px;height:56px;border-radius:50%;
                background:{GOLD};color:{INK};
                display:flex;align-items:center;justify-content:center;
                font-size:22px;font-weight:800;flex-shrink:0;
                box-shadow:0 4px 16px rgba(255,212,0,0.35);'>H</div>
            <div>
                <div style='font-size:18px;font-weight:700;color:{TEXT};'>Hamna Munir</div>
                <div style='font-size:13px;color:{MUTED};margin-top:3px;'>AI/ML Engineer · Software Engineering Student</div>
            </div>
        </div>
        {tags_html}
        """), unsafe_allow_html=True)

        # Problem statement
        st.markdown(card(f"""
        {section_title("🚨","Problem We Solve")}
        <div style='font-size:14px;color:{MUTED};line-height:1.9;'>
            Teachers cannot monitor 30+ students simultaneously. Students
            who lose focus go unnoticed until they fall behind. There was
            no tool combining:<br><br>
            <span style='color:{GOLD};font-weight:600;'>👁 Real-time attention tracking</span><br>
            <span style='color:{GOLD};font-weight:600;'>🎯 Per-student engagement scoring</span><br>
            <span style='color:{GOLD};font-weight:600;'>🤖 AI-generated teaching reports</span><br>
            <span style='color:{GOLD};font-weight:600;'>🔔 Automated distraction alerts</span>
        </div>
        """), unsafe_allow_html=True)

        # Project highlights — quick stat grid
        st.markdown(card(f"""
        {section_title("📌","Project Highlights")}
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
            <div style='text-align:center;background:{PANEL_ALT};border-radius:10px;padding:14px;'>
                <div class='smc-mono' style='font-size:24px;font-weight:800;color:{GOLD};'>7</div>
                <div style='font-size:11px;color:{MUTED};'>Emotions Tracked</div>
            </div>
            <div style='text-align:center;background:{PANEL_ALT};border-radius:10px;padding:14px;'>
                <div class='smc-mono' style='font-size:24px;font-weight:800;color:{GREEN};'>30+</div>
                <div style='font-size:11px;color:{MUTED};'>Students / Frame</div>
            </div>
            <div style='text-align:center;background:{PANEL_ALT};border-radius:10px;padding:14px;'>
                <div class='smc-mono' style='font-size:24px;font-weight:800;color:{GOLD};'>3</div>
                <div style='font-size:11px;color:{MUTED};'>Capture Modes</div>
            </div>
            <div style='text-align:center;background:{PANEL_ALT};border-radius:10px;padding:14px;'>
                <div class='smc-mono' style='font-size:24px;font-weight:800;color:{GREEN};'>0–100</div>
                <div style='font-size:11px;color:{MUTED};'>Engagement Scale</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Roadmap
        roadmap = [
            "Multi-camera classroom coverage",
            "Attendance auto-marking from face recognition",
            "Weekly trend reports emailed to teachers",
            "Mobile app for on-the-go monitoring",
        ]
        roadmap_html = section_title("🧭","Roadmap")
        for item in roadmap:
            roadmap_html += f"""
            <div style='display:flex;align-items:center;gap:10px;padding:8px 0;
                border-bottom:1px solid {BORDER_SOFT};'>
                <span style='color:{MUTED};font-weight:700;font-size:14px;flex-shrink:0;'>○</span>
                <span style='font-size:13px;color:{MUTED};'>{item}</span>
            </div>"""
        st.markdown(card(roadmap_html), unsafe_allow_html=True)

    with col2:
        # Tech stack
        tech_items = [
            ("🐍","Python 3.11","Core language"),
            ("🌊","Streamlit 1.38","Web dashboard"),
            ("👁","OpenCV 4.9","Computer vision"),
            ("🎯","YOLOv8 (Ultralytics)","Multi-face classroom detection"),
            ("🎯","MediaPipe 0.10","Face mesh & landmarks"),
            ("🧠","ONNX Runtime 1.20","Emotion model inference"),
            ("🌲","RandomForest","Focus detection model"),
            ("✦","Google Gemini 1.5","AI report generation"),
            ("📊","Plotly","Interactive charts"),
        ]
        tech_html = section_title("🛠️","Tech Stack")
        for icon, name, desc in tech_items:
            tech_html += f"""
            <div style='display:flex;align-items:center;gap:12px;padding:10px 12px;
                background:{PANEL_ALT};border-radius:8px;margin-bottom:6px;'>
                <span style='font-size:16px;'>{icon}</span>
                <div>
                    <div style='font-size:13px;font-weight:600;color:{TEXT};'>{name}</div>
                    <div style='font-size:11px;color:{MUTED};'>{desc}</div>
                </div>
            </div>"""
        st.markdown(card(tech_html), unsafe_allow_html=True)

        # Features
        features = [
            "Real-time emotion detection — 7 emotions",
            "Focus scoring 0–100 using eye landmarks",
            "YOLOv8-powered multi-face classroom detection (30+ students)",
            "Head pose & gaze direction analysis",
            "Photo, video & live snapshot input modes",
            "Per-student engagement grid view",
            "3 AI report types via Google Gemini",
            "Smart alerts when engagement drops",
            "Session history with CSV export",
            "Blink detection & attention monitoring",
        ]
        feat_html = section_title("✨","Key Features")
        for feat in features:
            feat_html += f"""
            <div style='display:flex;align-items:center;gap:10px;padding:8px 0;
                border-bottom:1px solid {BORDER_SOFT};'>
                <span style='color:{GOLD};font-weight:700;font-size:14px;flex-shrink:0;'>✓</span>
                <span style='font-size:13px;color:{MUTED};'>{feat}</span>
            </div>"""
        st.markdown(card(feat_html), unsafe_allow_html=True)

    # AI Models detail
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        # Emotion tags — built as a real loop before the card(), same fix
        # pattern as the developer tags above.
        emotions_html = "<div style='display:flex;gap:6px;flex-wrap:wrap;'>"
        for e in ["Happy","Sad","Angry","Neutral","Fear","Surprise","Disgust"]:
            emotions_html += f"<span style='background:{PANEL_ALT};color:{MUTED};border:1px solid {BORDER};padding:3px 10px;border-radius:6px;font-size:12px;'>{e}</span>"
        emotions_html += "</div>"

        st.markdown(card(f"""
        {section_title("🧠","Emotion Model")}
        <div style='background:{PANEL_ALT};border-radius:10px;padding:16px;margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div style='font-size:14px;font-weight:700;color:{TEXT};'>CNN → ONNX Runtime</div>
                <span style='background:{GOLD_SOFT};color:{GOLD};border:1px solid rgba(255,212,0,0.3);
                    padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>66.6% acc</span>
            </div>
            <div style='font-size:13px;color:{MUTED};line-height:1.8;'>
                Dataset: FER2013<br>
                Architecture: CNN<br>
                Format: ONNX (cross-platform)<br>
                Output: 7 emotion classes
            </div>
        </div>
        {emotions_html}
        """), unsafe_allow_html=True)

    with col4:
        # Focus-model feature tags — same fix pattern.
        focus_tags_html = "<div style='display:flex;gap:6px;flex-wrap:wrap;'>"
        for ft in ["Left EAR","Right EAR","Avg EAR","Eye Width L","Eye Width R","Eye Distance"]:
            focus_tags_html += f"<span style='background:{PANEL_ALT};color:{MUTED};border:1px solid {BORDER};padding:3px 10px;border-radius:6px;font-size:12px;'>{ft}</span>"
        focus_tags_html += "</div>"

        st.markdown(card(f"""
        {section_title("🎯","Focus Detection Model")}
        <div style='background:{PANEL_ALT};border-radius:10px;padding:16px;margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div style='font-size:14px;font-weight:700;color:{TEXT};'>RandomForest Classifier</div>
                <span style='background:{GREEN_SOFT};color:{GREEN};border:1px solid rgba(62,214,127,0.3);
                    padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>6 features</span>
            </div>
            <div style='font-size:13px;color:{MUTED};line-height:1.8;'>
                Classes: Focused (1) / Distracted (0)<br>
                Input: Eye landmarks via MediaPipe<br>
                Features: EAR, eye width, eye distance<br>
                Output: Focus probability 0–100%
            </div>
        </div>
        {focus_tags_html}
        """), unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(card(f"""
    {section_title("🏗️","System Architecture")}
    <div style='font-family:JetBrains Mono,monospace;font-size:12px;
        color:{MUTED};line-height:2;padding:8px;'>
        <span style='color:{GOLD};'>Input</span> (Photo / Video / Webcam)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GOLD};'>Face Detection — 3-stage fallback</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;1. <span style='color:{GREEN};'>YOLOv8</span> <span style='color:{MUTED};'>(small / angled / partial faces — classroom-grade)</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;2. <span style='color:{AMBER};'>OpenCV Cascade</span> <span style='color:{MUTED};'>+ CLAHE enhancement</span> <span style='color:{MUTED};'>— if YOLO finds 0 faces</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;3. <span style='color:{MUTED};'>MediaPipe full-image scan</span> <span style='color:{MUTED};'>— last resort</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GOLD};'>MediaPipe Face Mesh</span> → 468 landmarks per detected face<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GREEN};'>Emotion CNN</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{GREEN};'>Focus RandomForest</span><br>
        <span style='color:{MUTED};'>(7 emotions)</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{MUTED};'>(Focused / Distracted)</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GOLD};'>Engagement Score</span> (0–100) + <span style='color:{GOLD};'>State Classification</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GOLD};'>Agent Rules</span> → Smart Alerts<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GOLD};'>Google Gemini 1.5 Flash</span> → AI Reports<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style='color:{GREEN};'>Teacher Dashboard</span> (Streamlit)
    </div>
    """), unsafe_allow_html=True)

# ── FOOTER
st.markdown(f"""
<div style='text-align:center;padding:20px;color:{FAINT};font-size:12px;
    border-top:1px solid {BORDER_SOFT};margin-top:28px;'>
    🎓 SmartClass AI · Console Edition · Week 2 · Phase 1 &nbsp;·&nbsp;
    Built by <span style='color:{GOLD};font-weight:700;'>Hamna Munir</span>
    &nbsp;·&nbsp; 2026
</div>""", unsafe_allow_html=True)