"""
apps/streamlit/layout.py
========================
Generic system page renderer.

Architecture: SystemSpec + render_system()
------------------------------------------
Each dynamical system is described by a SystemSpec dataclass.
render_system() is a template that handles all the Streamlit plumbing:

    1. Render controls (sidebar) via spec.controls(prefix)
    2. On "Run" click  →  spec.run(controls) → (cfg, out)
    3. Cache (cfg, out) in st.session_state under "{prefix}_cfg" / "{prefix}_out"
       so results survive widget interactions without re-running the ODE solver.
    4. Render output via spec.make_dashboard(cfg, out, controls)

Two rendering modes
-------------------
  make_dashboard  (preferred):
      Single Plotly figure with Frames for synced animation + plots.
      All sub-charts share one Play/Pause button.

  make_animation + plots  (fallback):
      Separate animation figure left, stacked plot panels right.
      No cross-chart frame sync.

How to add a new system
-----------------------
1. Create apps/streamlit/systems/my_system_view.py
2. Implement: controls(prefix), run(controls), make_dashboard(cfg, out, ui)
3. Return a SystemSpec from get_spec()
4. Register in apps/streamlit/registry.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple, Optional, List

import streamlit as st
import plotly.graph_objects as go

# Type aliases for the four function signatures used in SystemSpec
Controls = Dict[str, Any]
Cfg = Dict[str, Any]
Out = Dict[str, Any]

ControlsFn = Callable[[str], Controls]          # (prefix) -> controls dict (must include run_clicked)
RunFn = Callable[[Controls], Tuple[Cfg, Out]]   # (controls) -> (cfg, out)
FigFn = Callable[[Cfg, Out, Controls], go.Figure]
CaptionFn = Callable[[Cfg, Out], str]


@dataclass
class PlotPanel:
    """A single titled plot in the fallback (non-dashboard) layout."""
    title: str
    make_fig: FigFn
    height: int = 220


@dataclass
class SystemSpec:
    """Descriptor for one dynamical system page.

    Fields
    ------
    id : str
        Short unique identifier used to namespace session_state keys (e.g. "ip", "sp").
    title : str
        Human-readable title shown in the UI.
    controls : ControlsFn
        Renders sidebar widgets and returns a dict of parameter values.
        Must include key "run_clicked" (bool).
    run : RunFn
        Builds DSS model, runs ODE solver, returns (cfg, out) dicts.
    caption : CaptionFn, optional
        Short text shown above the dashboard (e.g. "Δt ≈ 0.01 s · N = 1001").
    make_dashboard : FigFn, optional
        Preferred: returns a single Plotly figure with animation frames.
    make_animation : FigFn, optional
        Fallback: animated figure only (left column).
    plots : list[PlotPanel]
        Fallback: additional plot panels (right column).
    """
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
    """Generic page renderer — call this from the main app with a SystemSpec.

    Parameters
    ----------
    spec : SystemSpec
        Descriptor for the system to render.
    controls_container : st container, optional
        Where to place controls (default: st.sidebar).
    content_container : st container, optional
        Where to place output (default: main page).
    """
    prefix = spec.id   # used to namespace all session_state keys for this system
    if controls_container is None:
        controls_container = st.sidebar
    if content_container is None:
        content_container = st

    # ------------------------------------------------------------------
    # 1. Render controls and capture widget values
    # ------------------------------------------------------------------
    with controls_container:
        with st.container(key="controls", gap=None):
            controls = spec.controls(prefix)

    controls_raw = controls
    run_clicked = bool(controls_raw.get("run_clicked", False))

    # Remove run_clicked from the dict passed to run() / make_dashboard()
    controls = dict(controls_raw)
    controls.pop("run_clicked", None)

    # ------------------------------------------------------------------
    # 2. Session-state keys for cached results
    #    Results persist across widget interactions without re-running the solver.
    # ------------------------------------------------------------------
    cfg_key = f"{prefix}_cfg"
    out_key = f"{prefix}_out"
    # Cached Plotly figures (dashboard and fallback) survive reruns that don't
    # change the simulation output or the visualisation-relevant settings.
    fig_key = f"{prefix}_fig"
    fig_anim_key = f"{prefix}_fig_anim"
    fig_plots_key = f"{prefix}_fig_plots"
    fig_hash_key = f"{prefix}_fig_hash"
    st.session_state.setdefault(cfg_key, None)
    st.session_state.setdefault(out_key, None)
    st.session_state.setdefault(fig_key, None)
    st.session_state.setdefault(fig_anim_key, None)
    st.session_state.setdefault(fig_plots_key, None)
    st.session_state.setdefault(fig_hash_key, None)

    # ------------------------------------------------------------------
    # 3. Run simulation when the user clicks "Run"
    # ------------------------------------------------------------------
    if run_clicked:
        with st.spinner("Running simulation..."):
            cfg, out = spec.run(controls)
        st.session_state[cfg_key] = cfg
        st.session_state[out_key] = out
        # Invalidate cached figures so they are rebuilt from the new results.
        st.session_state[fig_key] = None
        st.session_state[fig_anim_key] = None
        st.session_state[fig_plots_key] = None
        st.session_state[fig_hash_key] = None

    cfg = st.session_state[cfg_key]
    out = st.session_state[out_key]

    # ------------------------------------------------------------------
    # 4. Render output
    # ------------------------------------------------------------------
    with content_container:
        if cfg is None or out is None:
            st.info("Configure parameters on the left and click Run.")
            return

        # Optional one-line caption (e.g. Δt, N, solver info)
        if spec.caption is not None:
            st.caption(spec.caption(cfg, out))

        # Visualisation settings that actually affect how the figure looks.
        # Any change here (but not to physics params) triggers a figure rebuild.
        _VIZ_KEYS = {"fps_anim", "max_frames", "max_plot_pts", "trail_on", "trail_max_points", "live_plots"}
        viz_hash = hash(str(sorted((k, controls.get(k)) for k in _VIZ_KEYS)))

        # --- Dashboard mode (preferred): single Plotly figure with frames ---
        if spec.make_dashboard is not None:
            if st.session_state[fig_key] is None or st.session_state[fig_hash_key] != viz_hash:
                st.session_state[fig_key] = spec.make_dashboard(cfg, out, controls)
                st.session_state[fig_hash_key] = viz_hash
            st.plotly_chart(st.session_state[fig_key], width='stretch', config={"displayModeBar": False})
            return

        # --- Fallback mode: animation left, plot stack right ---
        col_anim, col_plots = st.columns([1.25, 1.0], gap="large")

        if spec.make_animation is not None:
            with col_anim:
                if st.session_state[fig_anim_key] is None or st.session_state[fig_hash_key] != viz_hash:
                    st.session_state[fig_anim_key] = spec.make_animation(cfg, out, controls)
                    st.session_state[fig_hash_key] = viz_hash
                st.plotly_chart(st.session_state[fig_anim_key], width='stretch', config={"displayModeBar": False})

        with col_plots:
            cached_plots = st.session_state[fig_plots_key]
            if cached_plots is None or st.session_state[fig_hash_key] != viz_hash:
                cached_plots = []
                for p in spec.plots:
                    figp = p.make_fig(cfg, out, controls)
                    figp.update_layout(height=p.height, margin=dict(l=8, r=8, t=25, b=8))
                    cached_plots.append((p.title, figp))
                st.session_state[fig_plots_key] = cached_plots
                st.session_state[fig_hash_key] = viz_hash
            for title, figp in cached_plots:
                st.markdown(f"**{title}**")
                st.plotly_chart(figp, width='stretch', config={"displayModeBar": False})
