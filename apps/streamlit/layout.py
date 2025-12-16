from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple, Optional, List

import streamlit as st
import plotly.graph_objects as go

Controls = Dict[str, Any]
Cfg = Dict[str, Any]
Out = Dict[str, Any]

ControlsFn = Callable[[str], Controls]          # (prefix) -> controls dict (must include run_clicked)
RunFn = Callable[[Controls], Tuple[Cfg, Out]]   # (controls) -> (cfg, out)
FigFn = Callable[[Cfg, Out, Controls], go.Figure]
CaptionFn = Callable[[Cfg, Out], str]


@dataclass
class PlotPanel:
    title: str
    make_fig: FigFn
    height: int = 220


@dataclass
class SystemSpec:
    id: str
    title: str
    controls: ControlsFn
    run: RunFn
    caption: Optional[CaptionFn] = None

    # Preferred: a single dashboard figure (Plotly frames drive both animation + plots)
    make_dashboard: Optional[FigFn] = None

    # Alternative: separate animation + right stacked plots (no Plotly Play sync across charts)
    make_animation: Optional[FigFn] = None
    plots: List[PlotPanel] = field(default_factory=list)


def render_system(
    spec: SystemSpec,
    *,
    controls_container: Optional[st.delta_generator.DeltaGenerator] = None,
    content_container: Optional[st.delta_generator.DeltaGenerator] = None,
) -> None:
    """Generic page renderer.

    - Controls render into `controls_container` (default: st.sidebar).
    - Output renders into `content_container` (default: main page).
    - State is namespaced by `spec.id`.
    """
    prefix = spec.id
    if controls_container is None:
        controls_container = st.sidebar
    if content_container is None:
        content_container = st

    # Controls
    with controls_container:
        with st.container(key="controls", gap=None):
            controls = spec.controls(prefix)

    controls_raw = controls

    run_clicked = bool(controls_raw.get("run_clicked", False))

    controls = dict(controls_raw)

    controls.pop("run_clicked", None)
    # Session state
    cfg_key = f"{prefix}_cfg"
    out_key = f"{prefix}_out"
    st.session_state.setdefault(cfg_key, None)
    st.session_state.setdefault(out_key, None)

    if run_clicked:
        cfg, out = spec.run(controls)
        st.session_state[cfg_key] = cfg
        st.session_state[out_key] = out

    cfg = st.session_state[cfg_key]
    out = st.session_state[out_key]

    # Output
    with content_container:
        if cfg is None or out is None:
            st.info("Configure parameters on the left and click Run.")
            return

        if spec.caption is not None:
            st.caption(spec.caption(cfg, out))

        if spec.make_dashboard is not None:
            fig = spec.make_dashboard(cfg, out, controls)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            return

        # Fallback mode: separate animation + plot stack
        col_anim, col_plots = st.columns([1.25, 1.0], gap="large")

        if spec.make_animation is not None:
            with col_anim:
                fig_anim = spec.make_animation(cfg, out, controls)
                st.plotly_chart(fig_anim, width='stretch', config={"displayModeBar": False})

        with col_plots:
            for p in spec.plots:
                st.markdown(f"**{p.title}**")
                figp = p.make_fig(cfg, out, controls)
                figp.update_layout(height=p.height, margin=dict(l=8, r=8, t=25, b=8))
                st.plotly_chart(figp, width='stretch', config={"displayModeBar": False})
