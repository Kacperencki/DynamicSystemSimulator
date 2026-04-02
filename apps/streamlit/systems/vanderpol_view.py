from __future__ import annotations

from typing import Any, Dict, Tuple

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
from apps.streamlit.runners.vanderpol_runner import run_vanderpol
from apps.streamlit.components.dashboards.vanderpol_dashboard import make_vanderpol_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "L", "C", "mu",
    "v0", "iL0",
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": dict(
        L=1.0, C=1.0, mu=2.0,
        v0=1.0, iL0=0.0,
        t0=0.0, t1=20.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-7,
        fps_anim=30, max_frames=400, max_plot_pts=2500,
        trail_on=True, trail_max_points=220,
    ),
    "Relaxation oscillation": dict(
        L=1.0, C=1.0, mu=8.0,
        v0=0.2, iL0=0.0,
        t0=0.0, t1=40.0, dt=0.01,
        solver_method="Radau", rtol=1e-4, atol=1e-7,
        fps_anim=30, max_frames=400, max_plot_pts=3000,
        trail_on=True, trail_max_points=240,
    ),
    "Slow limit-cycle approach": dict(
        # μ=0.3: oscillator is still nearly harmonic but amplitude drifts noticeably
        # outward from v0=0.3 toward the limit cycle (~2 V) over ~80 s (≈3 time constants).
        # Longer than usual by necessity — weak nonlinearity simply takes time to act.
        L=1.0, C=1.0, mu=0.3,
        v0=0.3, iL0=0.0,
        t0=0.0, t1=80.0, dt=0.05,
        solver_method="RK45", rtol=1e-4, atol=1e-7,
        fps_anim=30, max_frames=400, max_plot_pts=2500,
        trail_on=True, trail_max_points=260,
    ),
    "Amplitude collapse": dict(
        # Start at v0=4.5 V, well outside the limit cycle (~2 V amplitude).
        # The nonlinear damping is large for |v|>1, so the envelope decays quickly
        # until the trajectory locks onto the attractor from the outside.
        L=1.0, C=1.0, mu=2.0,
        v0=4.5, iL0=0.0,
        t0=0.0, t1=20.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-7,
        fps_anim=30, max_frames=400, max_plot_pts=2500,
        trail_on=True, trail_max_points=240,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Default")

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Reset", default_preset=PRESETS.get("Default", {}), default_preset_name="Default")

        with st.expander("System parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                L = st.number_input("L [H]", value=1.0, min_value=1e-9, format="%.6g", key=f"{prefix}_L",
                                    help="Inductance [H]. Scales the inductor time constant.")
            with c2:
                C = st.number_input("C [F]", value=1.0, min_value=1e-9, format="%.6g", key=f"{prefix}_C",
                                    help="Capacitance [F]. Scales the oscillation frequency.")
            with c3:
                mu = st.number_input("μ", value=2.0, min_value=0.0, format="%.6g", key=f"{prefix}_mu",
                                     help="Nonlinear damping strength. μ=0: harmonic oscillator; μ>>1: relaxation oscillation.")

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                v0 = st.number_input("v(0)", value=1.0, format="%.6g", key=f"{prefix}_v0", help="Initial capacitor voltage (v).")
            with c2:
                iL0 = st.number_input("i_L(0)", value=0.0, format="%.6g", key=f"{prefix}_iL0", help="Initial inductor current (i_L).")

        t0, t1, dt = simulation_time(prefix, expanded=False, t0_default=0.0, t1_default=30.0, dt_default=0.01, dt_min=1e-5, dt_step=0.001, dt_format="%.6f")

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 200, 1200, 600, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 600, 12000, 5000, 200),
            trail_default=True,
            trail_checkbox_label="Show trajectory trail",
            trail_max_points=SliderSpec("Trail max points", 50, 600, 260, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

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
        **solver,
        **perf,
        save_run=save_run,
        log_dir=log_dir,
        run_name=run_name,
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(L=controls["L"], C=controls["C"], mu=controls["mu"])
    ic = dict(v0=controls["v0"], iL0=controls["iL0"])
    return run_vanderpol(
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
    return f"N = {len(out['T'])}"


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
