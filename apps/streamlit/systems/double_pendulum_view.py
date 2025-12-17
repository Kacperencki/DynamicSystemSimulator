from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.components.ui_sections import (
    SliderSpec,
    presets_selector,
    run_clear_row_form,
    simulation_time,
    solver_settings,
    animation_performance,
    logging_settings,
)
from apps.streamlit.runners.pendulum_runner import run_double_pendulum
from apps.streamlit.components.dashboards.double_pendulum_dashboard import make_double_pendulum_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "mode",
    "l1", "m1", "l2", "m2", "g",
    "b1", "b2", "fc1", "fc2",
    "A1", "w1", "phi1", "A2", "w2", "phi2",
    "th1_0", "w1_0", "th2_0", "w2_0",
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": dict(
        mode="ideal",
        l1=1.0, m1=1.0,
        l2=1.0, m2=1.0,
        g=9.81,
        b1=0.0, b2=0.0, fc1=0.0, fc2=0.0,
        A1=0.0, w1=0.0, phi1=0.0,
        A2=0.0, w2=0.0, phi2=0.0,
        th1_0=float(np.pi / 2), w1_0=0.0,
        th2_0=float(np.pi / 2), w2_0=0.0,
        t0=0.0, t1=12.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=360, max_plot_pts=2500,
        trail_on=False, trail_max_points=180,
    ),
    "Chaotic (ideal)": dict(
        mode="ideal",
        l1=1.0, m1=1.0,
        l2=1.0, m2=1.0,
        g=9.81,
        b1=0.0, b2=0.0, fc1=0.0, fc2=0.0,
        A1=0.0, w1=0.0, phi1=0.0,
        A2=0.0, w2=0.0, phi2=0.0,
        th1_0=float(0.9 * np.pi), w1_0=0.0,
        th2_0=float(0.95 * np.pi), w2_0=0.0,
        t0=0.0, t1=20.0, dt=0.01,
        solver_method="RK45", rtol=1e-5, atol=1e-7,
        fps_anim=30, max_frames=520, max_plot_pts=4000,
        trail_on=True, trail_max_points=240,
    ),
    "Damped": dict(
        mode="damped",
        l1=1.0, m1=1.0,
        l2=1.0, m2=1.0,
        g=9.81,
        b1=0.05, b2=0.05, fc1=0.01, fc2=0.01,
        A1=0.0, w1=0.0, phi1=0.0,
        A2=0.0, w2=0.0, phi2=0.0,
        th1_0=float(np.pi / 2), w1_0=0.0,
        th2_0=float(np.pi / 2), w2_0=0.0,
        t0=0.0, t1=14.0, dt=0.01,
        solver_method="Radau", rtol=1e-5, atol=1e-7,
        fps_anim=30, max_frames=420, max_plot_pts=3000,
        trail_on=False, trail_max_points=180,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Default")

    mode = st.selectbox(
        "Mode",
        ["ideal", "damped", "driven"],
        index=0,
        key=f"{prefix}_mode",
    )

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Clear")

        with st.expander("Physical parameters", expanded=False):
            r1, r2, r3 = st.columns(3)
            with r1:
                l1 = st.number_input("l₁ [m]", value=1.0, min_value=0.0, key=f"{prefix}_l1")
                m1 = st.number_input("m₁ [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m1")
            with r2:
                l2 = st.number_input("l₂ [m]", value=1.0, min_value=0.0, key=f"{prefix}_l2")
                m2 = st.number_input("m₂ [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m2")
            with r3:
                g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g")

        b1 = float(st.session_state.get(f"{prefix}_b1", 0.02))
        b2 = float(st.session_state.get(f"{prefix}_b2", 0.02))
        fc1 = float(st.session_state.get(f"{prefix}_fc1", 0.0))
        fc2 = float(st.session_state.get(f"{prefix}_fc2", 0.0))
        A1 = float(st.session_state.get(f"{prefix}_A1", 0.0))
        w1 = float(st.session_state.get(f"{prefix}_w1", 0.0))
        phi1 = float(st.session_state.get(f"{prefix}_phi1", 0.0))
        A2 = float(st.session_state.get(f"{prefix}_A2", 0.0))
        w2 = float(st.session_state.get(f"{prefix}_w2", 0.0))
        phi2 = float(st.session_state.get(f"{prefix}_phi2", 0.0))

        if mode in ("damped", "driven"):
            with st.expander("Damping / friction", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    b1 = st.number_input("b₁ [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b1")
                    fc1 = st.number_input("F_c1 [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc1")
                with c2:
                    b2 = st.number_input("b₂ [N·m·s]", value=0.02, min_value=0.0, key=f"{prefix}_b2")
                    fc2 = st.number_input("F_c2 [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_fc2")

        if mode == "driven":
            with st.expander("Drive parameters", expanded=False):
                p1, p2, p3 = st.columns(3)
                with p1:
                    A1 = st.number_input("A₁ [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_A1")
                    A2 = st.number_input("A₂ [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_A2")
                with p2:
                    w1 = st.number_input("ω₁ [rad/s]", value=0.0, min_value=0.0, key=f"{prefix}_w1")
                    w2 = st.number_input("ω₂ [rad/s]", value=0.0, min_value=0.0, key=f"{prefix}_w2")
                with p3:
                    phi1 = st.number_input("φ₁ [rad]", value=0.0, key=f"{prefix}_phi1")
                    phi2 = st.number_input("φ₂ [rad]", value=0.0, key=f"{prefix}_phi2")
        else:
            A1 = A2 = w1 = w2 = phi1 = phi2 = 0.0

        if mode == "ideal":
            b1 = b2 = fc1 = fc2 = 0.0

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                th1_0 = st.number_input("θ₁₀ [rad]", value=float(np.pi / 2), min_value=-float(np.pi), max_value=float(np.pi), step=0.01, key=f"{prefix}_th1_0")
                w1_0 = st.number_input("ω₁₀ [rad/s]", value=0.0, min_value=-10.0, max_value=10.0, step=0.1, key=f"{prefix}_w1_0")
            with c2:
                th2_0 = st.number_input("θ₂₀ [rad]", value=float(np.pi / 2), min_value=-float(np.pi), max_value=float(np.pi), step=0.01, key=f"{prefix}_th2_0")
                w2_0 = st.number_input("ω₂₀ [rad/s]", value=0.0, min_value=-10.0, max_value=10.0, step=0.1, key=f"{prefix}_w2_0")

        t0, t1, dt = simulation_time(prefix, expanded=False, t0_default=0.0, t1_default=12.0, dt_default=0.01, dt_min=1e-5, dt_step=0.001, dt_format="%.6f")

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 120, 900, 420, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 600, 8000, 3000, 200),
            trail_default=False,
            trail_checkbox_label="Show tip trail",
            trail_max_points=SliderSpec("Trail max points", 50, 600, 200, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

    return dict(
        run_clicked=run_clicked,
        mode=str(mode),
        l1=float(l1),
        m1=float(m1),
        l2=float(l2),
        m2=float(m2),
        g=float(g),
        b1=float(b1),
        b2=float(b2),
        fc1=float(fc1),
        fc2=float(fc2),
        A1=float(A1),
        w1=float(w1),
        phi1=float(phi1),
        A2=float(A2),
        w2=float(w2),
        phi2=float(phi2),
        th1_0=float(th1_0),
        w1_0=float(w1_0),
        th2_0=float(th2_0),
        w2_0=float(w2_0),
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
        **solver,
        **perf,
        save_run=save_run,
        log_dir=log_dir,
        run_name=run_name,
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
    return run_double_pendulum(
        params,
        ic,
        controls["t0"],
        controls["t1"],
        controls["dt"],
        method=controls["solver_method"],
        rtol=controls["rtol"],
        atol=controls["atol"],
        save_run=bool(controls.get('save_run', False)),
        log_dir=str(controls.get('log_dir', 'logs')),
        run_name=str(controls.get('run_name', '')),
    )


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
