from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Dynamic System Simulator", layout="wide")

css_path = Path(__file__).resolve().parent / "apps" / "streamlit" / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

from apps.streamlit.layout import render_system
from apps.streamlit.registry import SYSTEM_FACTORIES, SYSTEM_LEGACY

# Main layout: compact controls (left) + dashboard (right)
col_controls, col_main = st.columns([0.8, 4], gap="small")

with col_controls:
    model_name = st.selectbox(
        "Model",
        list(SYSTEM_FACTORIES.keys()) + list(SYSTEM_LEGACY.keys()),
        key="model_select",
    )

if model_name in SYSTEM_FACTORIES:
    spec = SYSTEM_FACTORIES[model_name]()
    render_system(
        spec,
        controls_container=col_controls,
        content_container=col_main,
    )
else:
    with col_main:
        SYSTEM_LEGACY[model_name]()
