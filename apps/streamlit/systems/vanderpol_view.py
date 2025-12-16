from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.vanderpol_runner import run_vanderpol
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.vanderpol_dashboard import make_vanderpol_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "L", "C", "mu",
    "v0", "iL0",
    "t0", "t1", "dt",
    "fps_anim", "max_frames",
    "trail_on", "trail_max_points",
    "max_plot_pts",
]


def controls(prefix: str) -> Controls:
    c1, c2 = st.columns([1, 1])
    with c1:
        run_clicked = st.button("Run", key=f"{prefix}_run", type="primary", width='stretch')
    with c2:
        reset_defaults_button(prefix, RESET_KEYS, label="Reset")

    with st.expander("Circuit parameters", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            L = st.number_input("L [H]", value=1.0, min_value=0.0, key=f"{prefix}_L")
        with r2:
            C = st.number_input("C [F]", value=1.0, min_value=0.0, key=f"{prefix}_C")
        with r3:
            mu = st.number_input("μ", value=1.0, key=f"{prefix}_mu")

    with st.expander("Initial state", expanded=False):
        i1, i2 = st.columns(2)
        with i1:
            v0 = st.number_input("v(0) [V]", value=2.0, key=f"{prefix}_v0")
        with i2:
            iL0 = st.number_input("iL(0) [A]", value=0.0, key=f"{prefix}_iL0")

    with st.expander("Simulation time", expanded=False):
        t1c1, t1c2, t1c3 = st.columns(3)
        with t1c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with t1c2:
            t1 = st.number_input("t₁ [s]", value=40.0, min_value=0.0, key=f"{prefix}_t1")
        with t1c3:
            dt = st.number_input("Δt [s]", value=0.01, min_value=1e-5, step=0.001, format="%.6f", key=f"{prefix}_dt")

    with st.expander("Animation / performance", expanded=False):
        fps_anim = st.slider("Animation FPS", 10, 60, 30, 5, key=f"{prefix}_fps_anim")
        max_frames = st.slider("Max frames", 120, 800, 360, 20, key=f"{prefix}_max_frames")

        trail_on = st.checkbox("Show tail in phase portrait", value=False, key=f"{prefix}_trail_on")
        trail_max_points = st.slider("Tail points", 50, 800, 240, 10, key=f"{prefix}_trail_max_points")

        max_plot_pts = st.slider("Plot points (downsample)", 400, 8000, 2400, 200, key=f"{prefix}_max_plot_pts")

    return dict(
        run_clicked=run_clicked,
        L=float(L),
        C=float(C),
        mu=float(mu),
        v0=float(v0),
        iL0=float(iL0),
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
        fps_anim=int(fps_anim),
        max_frames=int(max_frames),
        trail_on=bool(trail_on),
        trail_max_points=int(trail_max_points),
        max_plot_pts=int(max_plot_pts),
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(L=controls["L"], C=controls["C"], mu=controls["mu"])
    ic = dict(v0=controls["v0"], iL0=controls["iL0"])
    return run_vanderpol(params, ic, controls["t0"], controls["t1"], controls["dt"])


def caption(cfg: Cfg, out: Out) -> str:
    T = np.asarray(out.get("T", []), dtype=float)
    dt_eff = float(np.mean(np.diff(T))) if len(T) > 1 else float(cfg.get("dt", 0.01))
    return f"Δt ≈ {dt_eff:.6f} s · N = {len(T)}"


def make_dashboard(cfg: Cfg, out: Out, ui: Controls):
    return make_vanderpol_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="vdp",
        title="Van der Pol oscillator",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
