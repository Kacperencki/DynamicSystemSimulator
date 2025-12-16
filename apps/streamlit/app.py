from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # project root (folder that contains "apps")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# MUST be first Streamlit call
st.set_page_config(page_title="Dynamic System Simulator", layout="wide")

# Load CSS (optional, safe if missing)
css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


from apps.streamlit.layout import render_system
from apps.streamlit.registry import SYSTEM_SPECS, SYSTEM_LEGACY

# Main layout: compact controls (left) + dashboard (right)
col_controls, col_main = st.columns([0.8, 4], gap="small")


with col_controls:
    st.markdown('<div id="controls-anchor"></div>', unsafe_allow_html=True)

    model_name = st.selectbox(
        "Model",
        list(SYSTEM_SPECS.keys()) + list(SYSTEM_LEGACY.keys()),
        key="model_select",
    )

if model_name in SYSTEM_SPECS:
    render_system(
        SYSTEM_SPECS[model_name],
        controls_container=col_controls,
        content_container=col_main,
    )
else:
    with col_main:
        SYSTEM_LEGACY[model_name]()
