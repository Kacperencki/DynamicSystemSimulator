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
from apps.streamlit.runners.dc_motor_runner import run_dc_motor
from apps.streamlit.components.dashboards.dc_motor_dashboard import make_dc_motor_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "R", "L", "Ke", "Kt", "J", "bm",
    "v_mode", "V0", "v_offset", "t_step", "v_freq", "v_duty",
    "load_mode", "tau_load", "b_load", "tau_c", "omega_eps",
    "i0", "omega0",
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": dict(
        R=2.0, L=1e-3, Ke=0.1, Kt=0.1, J=1e-3, bm=1e-4,
        v_mode="step", V0=12.0, v_offset=0.0, t_step=0.05, v_freq=2.0, v_duty=0.5,
        load_mode="none", tau_load=0.0, b_load=0.0, tau_c=0.0, omega_eps=0.5,
        i0=0.0, omega0=0.0,
        t0=0.0, t1=0.6, dt=0.001,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=600, max_plot_pts=3000,
        trail_on=False, trail_max_points=180,
    ),
    "Sine input": dict(
        R=2.0, L=1e-3, Ke=0.1, Kt=0.1, J=1e-3, bm=1e-4,
        v_mode="sine", V0=8.0, v_offset=6.0, t_step=0.05, v_freq=3.0, v_duty=0.5,
        load_mode="viscous", tau_load=0.0, b_load=2e-3, tau_c=0.0, omega_eps=0.5,
        i0=0.0, omega0=0.0,
        t0=0.0, t1=1.2, dt=0.001,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=700, max_plot_pts=5000,
        trail_on=False, trail_max_points=180,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Default")

    v_mode = st.selectbox(
        "Input voltage profile",
        ["constant", "step", "ramp", "sine", "square"],
        index=1,
        key=f"{prefix}_v_mode",
        help="Voltage waveform applied to the motor: constant DC, a step change, a ramp, a sine wave, or a square wave.",
    )

    load_mode = st.selectbox(
        "Load model",
        ["none", "constant", "viscous", "coulomb"],
        index=0,
        key=f"{prefix}_load_mode",
        help="none: no external load; constant: fixed torque; viscous: speed-proportional braking; coulomb: dry-friction load.",
    )

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Reset", default_preset=PRESETS.get("Default", {}), default_preset_name="Default")

        with st.expander("Motor parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                R = st.number_input("R [Ω]", value=2.0, min_value=0.0, format="%.6g", key=f"{prefix}_R", help="Armature resistance.")
                L = st.number_input("L [H]", value=1e-3, min_value=0.0, format="%.6g", key=f"{prefix}_L", help="Armature inductance.")
            with c2:
                Ke = st.number_input("Ke [V·s/rad]", value=0.1, min_value=0.0, format="%.6g", key=f"{prefix}_Ke", help="Back-EMF constant (voltage per unit angular speed).")
                Kt = st.number_input("Kt [N·m/A]", value=0.1, min_value=0.0, format="%.6g", key=f"{prefix}_Kt", help="Torque constant (torque per unit current).")
            with c3:
                J = st.number_input("J [kg·m²]", value=1e-3, min_value=0.0, format="%.6g", key=f"{prefix}_J", help="Rotor moment of inertia.")
                bm = st.number_input("bm [N·m·s]", value=1e-4, min_value=0.0, format="%.6g", key=f"{prefix}_bm", help="Viscous friction coefficient at the rotor.")

        with st.expander("Input voltage parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                V0 = st.number_input("V0 [V]", value=12.0, format="%.6g", key=f"{prefix}_V0", help="Peak (or constant) voltage amplitude.")
            with c2:
                v_offset = st.number_input("Offset [V]", value=0.0, format="%.6g", key=f"{prefix}_v_offset", help="DC offset added to the voltage waveform.")
            with c3:
                t_step = st.number_input("Step time [s]", value=0.05, min_value=0.0, format="%.6g", key=f"{prefix}_t_step", help="Time at which the step input switches on.")

            if v_mode in ("sine", "square"):
                d1, d2 = st.columns(2)
                with d1:
                    v_freq = st.number_input("f [Hz]", value=2.0, min_value=0.0, format="%.6g", key=f"{prefix}_v_freq", help="Frequency of the sine or square wave.")
                with d2:
                    v_duty = st.number_input("Duty (square)", value=0.5, min_value=0.0, max_value=1.0, format="%.3f", key=f"{prefix}_v_duty", help="Duty cycle of the square wave (0–1).")
            else:
                v_freq = float(st.session_state.get(f"{prefix}_v_freq", 2.0))
                v_duty = float(st.session_state.get(f"{prefix}_v_duty", 0.5))

        with st.expander("Load parameters", expanded=False):
            if load_mode == "none":
                tau_load = 0.0
                b_load = 0.0
                tau_c = 0.0
                omega_eps = float(st.session_state.get(f"{prefix}_omega_eps", 0.5))
            elif load_mode == "constant":
                tau_load = st.number_input("τ_load [N·m]", value=0.05, min_value=0.0, format="%.6g", key=f"{prefix}_tau_load", help="Constant resistive torque load on the shaft.")
                b_load = 0.0
                tau_c = 0.0
                omega_eps = float(st.session_state.get(f"{prefix}_omega_eps", 0.5))
            elif load_mode == "viscous":
                b_load = st.number_input("b_load [N·m·s]", value=2e-3, min_value=0.0, format="%.6g", key=f"{prefix}_b_load", help="Viscous load damping coefficient.")
                tau_load = 0.0
                tau_c = 0.0
                omega_eps = float(st.session_state.get(f"{prefix}_omega_eps", 0.5))
            else:  # coulomb
                tau_c = st.number_input("τ_c [N·m]", value=0.02, min_value=0.0, format="%.6g", key=f"{prefix}_tau_c", help="Coulomb (dry) friction torque of the load.")
                omega_eps = st.number_input("ω_eps [rad/s]", value=0.5, min_value=1e-6, format="%.6g", key=f"{prefix}_omega_eps", help="Speed threshold below which Coulomb friction is smoothed.")
                tau_load = 0.0
                b_load = 0.0

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                i0 = st.number_input("i(0) [A]", value=0.0, format="%.6g", key=f"{prefix}_i0", help="Initial armature current.")
            with c2:
                omega0 = st.number_input("ω(0) [rad/s]", value=0.0, format="%.6g", key=f"{prefix}_omega0", help="Initial angular speed.")

        t0, t1, dt = simulation_time(prefix, expanded=False, t0_default=0.0, t1_default=0.6, dt_default=0.001, dt_min=1e-6, dt_step=1e-4, dt_format="%.6f")

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 200, 1400, 600, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 600, 20000, 4000, 200),
            trail_default=False,
            trail_checkbox_label="Show trail",
            trail_max_points=SliderSpec("Trail max points", 50, 600, 200, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

    return dict(
        run_clicked=run_clicked,
        R=float(R),
        L=float(L),
        Ke=float(Ke),
        Kt=float(Kt),
        J=float(J),
        bm=float(bm),
        v_mode=str(v_mode),
        V0=float(V0),
        v_offset=float(v_offset),
        t_step=float(t_step),
        v_freq=float(v_freq),
        v_duty=float(v_duty),
        load_mode=str(load_mode),
        tau_load=float(tau_load),
        b_load=float(b_load),
        tau_c=float(tau_c),
        omega_eps=float(omega_eps),
        i0=float(i0),
        omega0=float(omega0),
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
        R=controls["R"],
        L=controls["L"],
        Ke=controls["Ke"],
        Kt=controls["Kt"],
        J=controls["J"],
        bm=controls["bm"],
        v_mode=controls["v_mode"],
        V0=controls["V0"],
        v_offset=controls["v_offset"],
        t_step=controls["t_step"],
        v_freq=controls["v_freq"],
        v_duty=controls["v_duty"],
        load_mode=controls["load_mode"],
        tau_load=controls["tau_load"],
        b_load=controls["b_load"],
        tau_c=controls["tau_c"],
        omega_eps=controls["omega_eps"],
    )
    ic = dict(i0=controls["i0"], omega0=controls["omega0"])
    return run_dc_motor(
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
    return make_dc_motor_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="dcm",
        title="DC motor",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
