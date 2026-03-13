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
from apps.streamlit.runners.lorenz_runner import run_lorenz
from apps.streamlit.components.dashboards.lorenz_dashboard import make_lorenz_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "sigma", "rho", "beta",
    "x0", "y0", "z0",
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Classic attractor": dict(
        sigma=10.0, rho=28.0, beta=8.0 / 3.0,
        x0=1.0, y0=0.0, z0=0.0,
        t0=0.0, t1=40.0, dt=0.005,
        solver_method="DOP853", rtol=1e-6, atol=1e-8,
        fps_anim=30, max_frames=700, max_plot_pts=8000,
        trail_on=True, trail_max_points=320,
    ),
    "Sensitive to IC": dict(
        sigma=10.0, rho=28.0, beta=8.0 / 3.0,
        x0=1.001, y0=0.0, z0=0.0,
        t0=0.0, t1=40.0, dt=0.005,
        solver_method="DOP853", rtol=1e-6, atol=1e-8,
        fps_anim=30, max_frames=700, max_plot_pts=8000,
        trail_on=True, trail_max_points=320,
    ),
    "Near-periodic (rho=99.96)": dict(
        sigma=10.0, rho=99.96, beta=8.0 / 3.0,
        x0=0.1, y0=0.0, z0=0.0,
        t0=0.0, t1=20.0, dt=0.005,
        solver_method="DOP853", rtol=1e-6, atol=1e-8,
        fps_anim=30, max_frames=500, max_plot_pts=5000,
        trail_on=True, trail_max_points=280,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Classic attractor")

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Reset", default_preset=PRESETS.get("Classic attractor", {}), default_preset_name="Classic attractor")

        with st.expander("System parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                sigma = st.number_input("σ", value=10.0, key=f"{prefix}_sigma", help="Prandtl number — ratio of momentum to thermal diffusivity. Controls the rate of velocity mixing.")
            with c2:
                rho = st.number_input("ρ", value=28.0, key=f"{prefix}_rho", help="Normalised Rayleigh number — controls the temperature difference driving convection. Chaos onset above ≈ 24.74.")
            with c3:
                beta = st.number_input("β", value=8.0 / 3.0, key=f"{prefix}_beta", help="Geometric factor related to the spatial frequency of convective rolls. Classic value: 8/3.")

        with st.expander("Initial state", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                x0 = st.number_input("x(0)", value=1.0, key=f"{prefix}_x0", help="Initial x state.")
            with c2:
                y0 = st.number_input("y(0)", value=1.0, key=f"{prefix}_y0", help="Initial y state.")
            with c3:
                z0 = st.number_input("z(0)", value=1.0, key=f"{prefix}_z0", help="Initial z state.")

        t0, t1, dt = simulation_time(prefix, expanded=False, t0_default=0.0, t1_default=40.0, dt_default=0.01, dt_min=1e-5, dt_step=0.001, dt_format="%.6f")

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 200, 1600, 700, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 800, 20000, 8000, 200),
            trail_default=True,
            trail_checkbox_label="Show tail (3D)",
            trail_max_points=SliderSpec("Tail max points", 50, 700, 280, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

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
        **solver,
        **perf,
        save_run=save_run,
        log_dir=log_dir,
        run_name=run_name,
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = dict(sigma=controls["sigma"], rho=controls["rho"], beta=controls["beta"])
    ic = dict(x0=controls["x0"], y0=controls["y0"], z0=controls["z0"])
    return run_lorenz(
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
    return make_lorenz_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="lor",
        title="Lorenz attractor",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
