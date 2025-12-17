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
from apps.streamlit.runners.pendulum_runner import run_single_pendulum
from apps.streamlit.components.dashboards.single_pendulum_dashboard import make_single_pendulum_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "mode",
    "L", "m", "g",
    "b", "fc",
    "A", "w", "phi",
    "theta0", "omega0",
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames",
    "trail_on", "trail_max_points",
    "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": dict(
        mode="ideal",
        L=1.0, m=1.0, g=9.81,
        b=0.0, fc=0.0,
        A=0.0, w=0.0, phi=0.0,
        theta0=float(np.pi / 3), omega0=0.0,
        t0=0.0, t1=10.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=360, max_plot_pts=2000,
        trail_on=False, trail_max_points=180,
    ),
    "Damped": dict(
        mode="damped",
        L=1.0, m=1.0, g=9.81,
        b=0.06, fc=0.01,
        A=0.0, w=0.0, phi=0.0,
        theta0=float(np.pi / 2), omega0=0.0,
        t0=0.0, t1=12.0, dt=0.01,
        solver_method="Radau", rtol=1e-5, atol=1e-7,
        fps_anim=30, max_frames=360, max_plot_pts=2200,
        trail_on=False, trail_max_points=180,
    ),
    "Driven": dict(
        mode="driven",
        L=1.0, m=1.0, g=9.81,
        b=0.02, fc=0.0,
        A=0.25, w=2.0, phi=0.0,
        theta0=0.15, omega0=0.0,
        t0=0.0, t1=20.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=520, max_plot_pts=3200,
        trail_on=True, trail_max_points=240,
    ),
    "DC-driven": dict(
        mode="dc_driven",
        L=1.0, m=1.0, g=9.81,
        b=0.02, fc=0.0,
        A=0.12, w=0.0, phi=0.0,
        theta0=0.3, omega0=0.0,
        t0=0.0, t1=15.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=420, max_plot_pts=2600,
        trail_on=False, trail_max_points=180,
    ),
}


def controls(prefix: str) -> Controls:
    # preset selector (reruns on change; applies to session_state)
    presets_selector(prefix, PRESETS, label="Preset", default_name="Default")

    # dynamic selector (outside the form so it can re-render dependent inputs)
    mode = st.selectbox(
        "Mode",
        ["ideal", "damped", "driven", "dc_driven"],
        index=0,
        key=f"{prefix}_mode",
    )

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Clear")

        with st.expander("Physical parameters", expanded=False):
            r1, r2, r3 = st.columns(3)
            with r1:
                L = st.number_input("L [m]", value=1.0, min_value=0.0, key=f"{prefix}_L")
            with r2:
                m = st.number_input("m [kg]", value=1.0, min_value=0.0, key=f"{prefix}_m")
            with r3:
                g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g")

        # defaults for conditional sections (keep their values between modes)
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
            A, w, phi = 0.0, 0.0, 0.0

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

            if mode == "dc_driven":
                w, phi = 0.0, 0.0

        else:
            b, fc, A, w, phi = 0.0, 0.0, 0.0, 0.0, 0.0

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                th0 = st.number_input(
                    "θ₀ [rad]",
                    value=float(np.pi / 3),
                    min_value=-float(np.pi),
                    max_value=float(np.pi),
                    step=0.01,
                    key=f"{prefix}_theta0",
                )
            with c2:
                w0 = st.number_input(
                    "ω₀ [rad/s]",
                    value=0.0,
                    min_value=-10.0,
                    max_value=10.0,
                    step=0.1,
                    key=f"{prefix}_omega0",
                )

        t0, t1, dt = simulation_time(
            prefix,
            expanded=False,
            t0_default=0.0,
            t1_default=10.0,
            dt_default=0.01,
            dt_min=1e-5,
            dt_step=0.001,
            dt_format="%.6f",
        )

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 120, 800, 360, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 400, 6000, 2000, 200),
            trail_default=False,
            trail_checkbox_label="Show tip trail",
            trail_max_points=SliderSpec("Trail max points", 50, 500, 180, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

    return dict(
        run_clicked=run_clicked,
        mode=str(mode),
        L=float(L),
        m=float(m),
        g=float(g),
        b=float(b),
        fc=float(fc),
        A=float(A),
        w=float(w),
        phi=float(phi),
        theta0=float(th0),
        omega0=float(w0),
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
    return run_single_pendulum(
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
