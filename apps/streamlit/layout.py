# apps/streamlit/layout.py

from typing import Callable, Dict, Any, Tuple
import streamlit as st

Controls = Dict[str, Any]
Cfg = Dict[str, Any]
Out = Dict[str, Any]

ControlsFn = Callable[[], Controls]
RunFn = Callable[[Controls], Tuple[Cfg, Out]]
RenderMainFn = Callable[[Cfg, Out], None]


def render_system_page(
    title: str,
    controls_panel: ControlsFn,
    run_simulation: RunFn,
    render_main: RenderMainFn,
    state_key: str,
) -> None:
    """
    Generic pattern used by ALL systems:

    - LEFT column : controls panel (sliders, inputs, 'Run simulation' button)
    - RIGHT column: animation + plots based on (cfg, out)

    controls_panel() MUST return a dict containing a boolean key 'run_clicked'.
    """

    st.subheader(title)

    # left = controls, right = main content
    # slightly narrower main column than before; large gap for breathing room
    col_ctrl, col_main = st.columns([1.1, 1.7], gap="large")

    # --- LEFT: controls + run button ---
    with col_ctrl:
        controls = controls_panel()

    run_clicked = bool(controls.pop("run_clicked", False))

    if run_clicked:
        # basic feedback when a simulation might take time
        with st.spinner("Running simulation..."):
            cfg, out = run_simulation(controls)
        st.session_state[f"{state_key}_cfg"] = cfg
        st.session_state[f"{state_key}_out"] = out

    cfg = st.session_state.get(f"{state_key}_cfg")
    out = st.session_state.get(f"{state_key}_out")

    # --- RIGHT: animation + plots ---
    with col_main:
        if cfg is None or out is None:
            # show a placeholder "canvas" so the right side does not look empty
            st.info("Configure parameters on the left and run the simulation.")
            st.markdown(
                "<div style='height:420px; background-color:#f9fafb; "
                "border-radius:0.5rem; border:1px dashed #e5e7eb;'></div>",
                unsafe_allow_html=True,
            )
            return

        render_main(cfg, out)
