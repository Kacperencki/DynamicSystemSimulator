# apps/streamlit/app.py
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.streamlit.views.single_pendulum_view import render_single_pendulum_page
from apps.streamlit.views.double_pendulum_view import render_double_pendulum_page
from apps.streamlit.views.vanderpool_view import render_vanderpol_page
from apps.streamlit.views.inverted_view import render_inverted_page


st.set_page_config(
    page_title="Dynamic System Simulator",
    layout="wide",
)

# -------------------- LEFT SIDEBAR: ONLY SYSTEM SELECT --------------------
with st.sidebar:
    system = st.selectbox(
        "System",
        [
            "Single pendulum",
            "Double pendulum",
            "Van der Pol oscillator",
            "Inverted pendulum / cart–pole",
        ],
        key="system_select",
    )

# -------------------- MAIN AREA (right side) ------------------------------
# No extra title / caption here — each view handles its own heading.

if system == "Single pendulum":
    render_single_pendulum_page()

elif system == "Double pendulum":
    render_double_pendulum_page()

elif system == "Van der Pol oscillator":
    render_vanderpol_page()

elif system == "Inverted pendulum / cart–pole":
    render_inverted_page()
