from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.pendulum_runner import run_double_pendulum
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.double_pendulum_dashboard import (
    make_double_pendulum_dashboard,
)

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "mode",
    "l1", "m1", "l2", "m2", "g",
    "b1", "b2", "fc1", "fc2",
    "A1", "w1", "phi1", "A2", "w2", "phi2",
    "th1_0", "w1_0", "th2_0", "w2_0",
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
        r1, r2 = st.columns(2)
        with r1:
            l1 = st.number_input("l₁ [m]", value=1.0, min_value=0.0, key=f"{prefix}_l1")
            m1 = st.number_input("m₁ [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m1")
        with r2:
            l2 = st.number_input("l₂ [m]", value=1.0, min_value=0.0, key=f"{prefix}_l2")
            m2 = st.number_input("m₂ [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m2")
        g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g")

    # Defaults (used when widgets are hidden by mode)
    b1 = float(st.session_state.get(f"{prefix}_b1", 0.02))
    b2 = float(st.session_state.get(f"{prefix}_b2", 0.02))
    fc1 = float(st.session_state.get(f"{prefix}_fc1", 0.0))
    fc2 = float(st.session_state.get(f"{prefix}_fc2", 0.0))
    A1 = float(st.session_state.get(f"{prefix}_A1", 0.1))
    w1 = float(st.session_state.get(f"{prefix}_w1", 2.0))
    phi1 = float(st.session_state.get(f"{prefix}_phi1", 0.0))
    A2 = float(st.session_state.get(f"{prefix}_A2", 0.1))
    w2 = float(st.session_state.get(f"{prefix}_w2", 2.0))
    phi2 = float(st.session_state.get(f"{prefix}_phi2", 0.0))

    is_damped = mode in ("damped", "driven", "dc_driven")
    is_driven = mode == "driven"

    if is_damped:
        with st.expander("Damping / friction", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                b1 = st.number_input("b₁ [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b1")
                fc1 = st.number_input("Coulomb F_c₁ [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc1")
            with d2:
                b2 = st.number_input("b₂ [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b2")
                fc2 = st.number_input("Coulomb F_c₂ [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc2")
    else:
        b1 = b2 = 0.0
        fc1 = fc2 = 0.0

    if is_driven:
        with st.expander("Drive torques", expanded=False):
            st.caption("Harmonic drives")
            p1, p2 = st.columns(2)
            with p1:
                A1 = st.number_input("A₁ [N·m]", value=0.1, min_value=0.0, key=f"{prefix}_A1")
                w1 = st.number_input("ω₁ [rad/s]", value=2.0, min_value=0.0, key=f"{prefix}_w1")
                phi1 = st.number_input("φ₁ [rad]", value=0.0, key=f"{prefix}_phi1")
            with p2:
                A2 = st.number_input("A₂ [N·m]", value=0.1, min_value=0.0, key=f"{prefix}_A2")
                w2 = st.number_input("ω₂ [rad/s]", value=2.0, min_value=0.0, key=f"{prefix}_w2")
                phi2 = st.number_input("φ₂ [rad]", value=0.0, key=f"{prefix}_phi2")
    else:
        A1 = A2 = 0.0
        w1 = w2 = 0.0
        phi1 = phi2 = 0.0

    with st.expander("Initial state", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            th1_0 = st.number_input(
                "θ₁₀ [rad]",
                value=float(np.pi / 2),
                min_value=-float(np.pi),
                max_value=float(np.pi),
                step=0.01,
                key=f"{prefix}_th1_0",
            )
            w1_0 = st.number_input(
                "ω₁₀ [rad/s]",
                value=0.0,
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key=f"{prefix}_w1_0",
            )
        with a2:
            th2_0 = st.number_input(
                "θ₂₀ [rad]",
                value=float(np.pi / 2 - 0.2),
                min_value=-float(np.pi),
                max_value=float(np.pi),
                step=0.01,
                key=f"{prefix}_th2_0",
            )
            w2_0 = st.number_input(
                "ω₂₀ [rad/s]",
                value=0.0,
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key=f"{prefix}_w2_0",
            )

    with st.expander("Simulation time", expanded=False):
        t1c1, t1c2, t1c3 = st.columns(3)
        with t1c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with t1c2:
            t1 = st.number_input("t₁ [s]", value=10.0, key=f"{prefix}_t1")
        with t1c3:
            dt = st.number_input(
                "Δt [s]",
                value=0.01,
                min_value=1e-5,
                step=0.001,
                format="%.6f",
                key=f"{prefix}_dt",
            )

    with st.expander("Animation / performance", expanded=False):
        fps_anim = st.slider("Animation FPS", 10, 60, 30, 5, key=f"{prefix}_fps_anim")
        max_frames = st.slider("Max frames", 120, 800, 360, 20, key=f"{prefix}_max_frames")

        trail_on = st.checkbox("Show tip trail", value=False, key=f"{prefix}_trail_on")
        trail_max_points = st.slider("Trail max points", 50, 600, 220, 10, key=f"{prefix}_trail_max_points")

        max_plot_pts = st.slider("Plot points (downsample)", 400, 8000, 2500, 250, key=f"{prefix}_max_plot_pts")

    return dict(
        run_clicked=run_clicked,
        mode=mode,
        l1=float(l1), m1=float(m1),
        l2=float(l2), m2=float(m2),
        g=float(g),
        b1=float(b1), b2=float(b2),
        fc1=float(fc1), fc2=float(fc2),
        A1=float(A1), w1=float(w1), phi1=float(phi1),
        A2=float(A2), w2=float(w2), phi2=float(phi2),
        th1_0=float(th1_0), w1_0=float(w1_0),
        th2_0=float(th2_0), w2_0=float(w2_0),
        t0=float(st.session_state.get(f"{prefix}_t0", 0.0)),
        t1=float(st.session_state.get(f"{prefix}_t1", 10.0)),
        dt=float(st.session_state.get(f"{prefix}_dt", 0.01)),
        fps_anim=int(st.session_state.get(f"{prefix}_fps_anim", 30)),
        max_frames=int(st.session_state.get(f"{prefix}_max_frames", 360)),
        trail_on=bool(st.session_state.get(f"{prefix}_trail_on", False)),
        trail_max_points=int(st.session_state.get(f"{prefix}_trail_max_points", 220)),
        max_plot_pts=int(st.session_state.get(f"{prefix}_max_plot_pts", 2500)),
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(
        mode=controls["mode"],
        l1=controls["l1"],
        m1=controls["m1"],
        l2=controls["l2"],
        m2=controls["m2"],
        g=controls["g"],
        b1=controls["b1"],
        b2=controls["b2"],
        fc1=controls["fc1"],
        fc2=controls["fc2"],
        A1=controls["A1"],
        w1=controls["w1"],
        phi1=controls["phi1"],
        A2=controls["A2"],
        w2=controls["w2"],
        phi2=controls["phi2"],
    )
    ic = dict(
        th1_0=controls["th1_0"],
        w1_0=controls["w1_0"],
        th2_0=controls["th2_0"],
        w2_0=controls["w2_0"],
    )
    return run_double_pendulum(params, ic, controls["t0"], controls["t1"], controls["dt"])


def caption(cfg: Cfg, out: Out) -> str:
    T = out["T"]
    dt_eff = float(np.mean(np.diff(T))) if len(T) > 1 else float(cfg.get("dt", 0.01))
    return f"Δt ≈ {dt_eff:.6f} s · N = {len(T)}"


def make_dashboard(cfg: Cfg, out: Out, ui: Controls):
    return make_double_pendulum_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="dp",
        title="Double pendulum",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
