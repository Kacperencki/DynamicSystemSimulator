from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.pendulum_runner import run_single_pendulum
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.single_pendulum_dashboard import make_single_pendulum_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "mode",
    "L", "m", "g",
    "b", "fc",
    "A", "w", "phi",
    "theta0", "omega0",
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

    mode = st.selectbox(
        "Mode",
        ["ideal", "damped", "driven", "dc_driven"],
        index=0,
        key=f"{prefix}_mode",
    )

    with st.expander("Physical parameters", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            L = st.number_input("L [m]", value=1.0, min_value=0.0, key=f"{prefix}_L")
        with r2:
            m = st.number_input("m [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m")
        with r3:
            g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g")

    b = float(st.session_state.get(f"{prefix}_b", 0.02))
    fc = float(st.session_state.get(f"{prefix}_fc", 0.0))
    A = float(st.session_state.get(f"{prefix}_A", 0.2))
    w = float(st.session_state.get(f"{prefix}_w", 2.0))
    phi = float(st.session_state.get(f"{prefix}_phi", 0.0))

    if mode == "damped":
        with st.expander("Damped mode parameters", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                b = st.number_input("b [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b")
            with d2:
                fc = st.number_input("Coulomb F_c [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc")
        A = 0.0
        w = 0.0
        phi = 0.0

    elif mode in ("driven", "dc_driven"):
        title = "Driven mode parameters" if mode == "driven" else "DC-driven mode parameters"
        with st.expander(title, expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                b = st.number_input("b [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b")
            with d2:
                fc = st.number_input("Coulomb F_c [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc")

            p1, p2, p3 = st.columns(3)
            with p1:
                A = st.number_input("A [N·m]", value=0.2, min_value=0.0, key=f"{prefix}_A")
            with p2:
                w = st.number_input("ω [rad/s]", value=2.0, min_value=0.0, key=f"{prefix}_w")
            with p3:
                phi = st.number_input("φ [rad]", value=0.0, key=f"{prefix}_phi")

    else:
        b = 0.0
        fc = 0.0
        A = 0.0
        w = 0.0
        phi = 0.0

    with st.expander("Initial state", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            th0 = st.number_input("θ₀ [rad]", value=float(np.pi/3), min_value=-float(np.pi), max_value=float(np.pi), step=0.01, key=f"{prefix}_theta0")
        with c2:
            w0 = st.number_input("ω₀ [rad/s]", value=0.0, min_value=-10.0, max_value=10.0, step=0.1, key=f"{prefix}_omega0")

    with st.expander("Simulation time", expanded=False):
        t1c1, t1c2, t1c3 = st.columns(3)
        with t1c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with t1c2:
            t1 = st.number_input("t₁ [s]", value=10.0, key=f"{prefix}_t1")
        with t1c3:
            dt = st.number_input("Δt [s]", value=0.01, min_value=1e-5, step=0.001, format="%.6f", key=f"{prefix}_dt")

    with st.expander("Animation / performance", expanded=False):
        fps_anim = st.slider("Animation FPS", 10, 60, 30, 5, key=f"{prefix}_fps_anim")
        max_frames = st.slider("Max frames", 120, 800, 360, 20, key=f"{prefix}_max_frames")

        trail_on = st.checkbox("Show tip trail", value=False, key=f"{prefix}_trail_on")
        trail_max_points = st.slider("Trail max points", 50, 500, 180, 10, key=f"{prefix}_trail_max_points")

        max_plot_pts = st.slider("Plot points (downsample)", 400, 6000, 2000, 200, key=f"{prefix}_max_plot_pts")

    return dict(
        run_clicked=run_clicked,
        mode=mode,
        L=L, m=m, g=g,
        b=b, fc=fc,
        A=A, w=w, phi=phi,
        theta0=float(th0), omega0=float(w0),
        t0=float(st.session_state.get(f"{prefix}_t0", 0.0)),
        t1=float(st.session_state.get(f"{prefix}_t1", 10.0)),
        dt=float(st.session_state.get(f"{prefix}_dt", 0.01)),
        fps_anim=int(st.session_state.get(f"{prefix}_fps_anim", 30)),
        max_frames=int(st.session_state.get(f"{prefix}_max_frames", 360)),
        trail_on=bool(st.session_state.get(f"{prefix}_trail_on", False)),
        trail_max_points=int(st.session_state.get(f"{prefix}_trail_max_points", 180)),
        max_plot_pts=int(st.session_state.get(f"{prefix}_max_plot_pts", 2000)),
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(
        mode=controls["mode"],
        L=controls["L"],
        m=controls["m"],
        g=controls["g"],
        b=controls["b"],
        fc=controls["fc"],
        A=controls["A"],
        w=controls["w"],
        phi=controls["phi"],
    )
    ic = dict(theta0=controls["theta0"], omega0=controls["omega0"])
    return run_single_pendulum(params, ic, controls["t0"], controls["t1"], controls["dt"])


def caption(cfg: Cfg, out: Out) -> str:
    T = out["T"]
    dt_eff = float(np.mean(np.diff(T))) if len(T) > 1 else float(cfg.get("dt", 0.01))
    return f"Δt ≈ {dt_eff:.6f} s · N = {len(T)}"


def make_dashboard(cfg: Cfg, out: Out, ui: Controls):
    return make_single_pendulum_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="sp",
        title="Single pendulum",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
