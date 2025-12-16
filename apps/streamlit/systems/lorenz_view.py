from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.lorenz_runner import run_lorenz
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.lorenz_dashboard import make_lorenz_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "sigma", "rho", "beta",
    "x0", "y0", "z0",
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

    with st.expander("System parameters", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            sigma = st.number_input("σ", value=10.0, min_value=0.0, key=f"{prefix}_sigma")
        with r2:
            rho = st.number_input("ρ", value=28.0, min_value=0.0, key=f"{prefix}_rho")
        with r3:
            beta = st.number_input("β", value=8.0 / 3.0, min_value=0.0, key=f"{prefix}_beta")

    with st.expander("Initial state", expanded=False):
        i1, i2, i3 = st.columns(3)
        with i1:
            x0 = st.number_input("x(0)", value=1.0, key=f"{prefix}_x0")
        with i2:
            y0 = st.number_input("y(0)", value=1.0, key=f"{prefix}_y0")
        with i3:
            z0 = st.number_input("z(0)", value=1.0, key=f"{prefix}_z0")

    with st.expander("Simulation time", expanded=False):
        t1c1, t1c2, t1c3 = st.columns(3)
        with t1c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with t1c2:
            t1 = st.number_input("t₁ [s]", value=25.0, min_value=0.0, key=f"{prefix}_t1")
        with t1c3:
            dt = st.number_input("Δt [s]", value=0.01, min_value=1e-5, step=0.001, format="%.6f", key=f"{prefix}_dt")

    with st.expander("Animation / performance", expanded=False):
        fps_anim = st.slider("Animation FPS", 10, 60, 30, 5, key=f"{prefix}_fps_anim")
        max_frames = st.slider("Max frames", 150, 900, 450, 25, key=f"{prefix}_max_frames")

        trail_on = st.checkbox("Show trajectory tail (instead of full)", value=True, key=f"{prefix}_trail_on")
        trail_max_points = st.slider("Tail points", 50, 1200, 350, 25, key=f"{prefix}_trail_max_points")

        max_plot_pts = st.slider("Plot points (downsample)", 800, 12000, 3000, 200, key=f"{prefix}_max_plot_pts")

    return dict(
        run_clicked=run_clicked,
        sigma=float(sigma),
        rho=float(rho),
        beta=float(beta),
        x0=float(x0),
        y0=float(y0),
        z0=float(z0),
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
    params = dict(sigma=controls["sigma"], rho=controls["rho"], beta=controls["beta"])
    ic = dict(x0=controls["x0"], y0=controls["y0"], z0=controls["z0"])
    return run_lorenz(params, ic, controls["t0"], controls["t1"], controls["dt"])


def caption(cfg: Cfg, out: Out) -> str:
    T = np.asarray(out.get("T", []), dtype=float)
    dt_eff = float(np.mean(np.diff(T))) if len(T) > 1 else float(cfg.get("dt", 0.01))
    return f"Δt ≈ {dt_eff:.6f} s · N = {len(T)}"


def make_dashboard(cfg: Cfg, out: Out, ui: Controls):
    return make_lorenz_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="lor",
        title="Lorenz system",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
