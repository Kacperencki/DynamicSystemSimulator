from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.dc_motor_runner import run_dc_motor
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.dc_motor_dashboard import make_dc_motor_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "R", "L", "Ke", "Kt", "J", "bm",
    "v_mode", "V0", "v_offset", "t_step", "v_freq", "v_duty",
    "load_mode", "tau_load", "b_load", "tau_c", "omega_eps",
    "i0", "omega0",
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

    with st.expander("Motor parameters", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            R = st.number_input("R [Ω]", value=1.0, min_value=1e-9, key=f"{prefix}_R")
        with r2:
            L = st.number_input("L [H]", value=0.5, min_value=1e-9, key=f"{prefix}_L")
        with r3:
            J = st.number_input("J [kg·m²]", value=0.01, min_value=1e-12, format="%.6f", key=f"{prefix}_J")

        k1, k2, k3 = st.columns(3)
        with k1:
            Ke = st.number_input("Kₑ [V/(rad/s)]", value=0.1, min_value=0.0, format="%.6f", key=f"{prefix}_Ke")
        with k2:
            Kt = st.number_input("Kₜ [N·m/A]", value=0.1, min_value=0.0, format="%.6f", key=f"{prefix}_Kt")
        with k3:
            bm = st.number_input("bₘ [N·m·s/rad]", value=0.001, min_value=0.0, format="%.6f", key=f"{prefix}_bm")

    v_mode = st.selectbox(
        "Voltage input",
        ["constant", "step", "sine", "square", "ramp"],
        index=1,
        key=f"{prefix}_v_mode",
    )

    V0 = float(st.session_state.get(f"{prefix}_V0", 6.0))
    v_offset = float(st.session_state.get(f"{prefix}_v_offset", 0.0))
    t_step = float(st.session_state.get(f"{prefix}_t_step", 0.05))
    v_freq = float(st.session_state.get(f"{prefix}_v_freq", 1.0))
    v_duty = float(st.session_state.get(f"{prefix}_v_duty", 0.5))

    with st.expander("Input parameters", expanded=False):
        cA, cB = st.columns(2)
        with cA:
            V0 = st.number_input("V₀ [V]", value=6.0, key=f"{prefix}_V0")
        with cB:
            v_offset = st.number_input("Offset [V]", value=0.0, key=f"{prefix}_v_offset")

        if v_mode in ("step", "ramp"):
            t_step = st.number_input("t_step [s]", value=0.05, min_value=0.0, format="%.4f", key=f"{prefix}_t_step")
            v_freq = float(st.session_state.get(f"{prefix}_v_freq", 1.0))
            v_duty = float(st.session_state.get(f"{prefix}_v_duty", 0.5))
        elif v_mode in ("sine", "square"):
            c1, c2 = st.columns(2)
            with c1:
                v_freq = st.number_input("f [Hz]", value=1.0, min_value=0.0, format="%.4f", key=f"{prefix}_v_freq")
            with c2:
                v_duty = st.slider("Duty (square)", 0.05, 0.95, 0.5, 0.05, key=f"{prefix}_v_duty")
            t_step = float(st.session_state.get(f"{prefix}_t_step", 0.05))
        else:
            t_step = float(st.session_state.get(f"{prefix}_t_step", 0.05))
            v_freq = float(st.session_state.get(f"{prefix}_v_freq", 1.0))
            v_duty = float(st.session_state.get(f"{prefix}_v_duty", 0.5))

    load_mode = st.selectbox(
        "Load torque",
        ["none", "constant", "viscous", "coulomb"],
        index=0,
        key=f"{prefix}_load_mode",
    )

    tau_load = float(st.session_state.get(f"{prefix}_tau_load", 0.0))
    b_load = float(st.session_state.get(f"{prefix}_b_load", 0.0))
    tau_c = float(st.session_state.get(f"{prefix}_tau_c", 0.0))
    omega_eps = float(st.session_state.get(f"{prefix}_omega_eps", 0.5))

    with st.expander("Load parameters", expanded=False):
        if load_mode == "constant":
            tau_load = st.number_input("τ_load [N·m]", value=0.02, format="%.6f", key=f"{prefix}_tau_load")
        elif load_mode == "viscous":
            b_load = st.number_input("b_load [N·m·s/rad]", value=0.002, min_value=0.0, format="%.6f", key=f"{prefix}_b_load")
        elif load_mode == "coulomb":
            c1, c2 = st.columns(2)
            with c1:
                tau_c = st.number_input("τ_c [N·m]", value=0.02, min_value=0.0, format="%.6f", key=f"{prefix}_tau_c")
            with c2:
                omega_eps = st.number_input("ω_smooth [rad/s]", value=0.5, min_value=1e-6, format="%.4f", key=f"{prefix}_omega_eps")

    with st.expander("Initial state", expanded=False):
        i1, i2 = st.columns(2)
        with i1:
            i0 = st.number_input("i(0) [A]", value=0.0, key=f"{prefix}_i0")
        with i2:
            omega0 = st.number_input("ω(0) [rad/s]", value=0.0, key=f"{prefix}_omega0")

    with st.expander("Simulation time", expanded=False):
        t1c1, t1c2, t1c3 = st.columns(3)
        with t1c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with t1c2:
            t1 = st.number_input("t₁ [s]", value=2.0, min_value=0.0, key=f"{prefix}_t1")
        with t1c3:
            dt = st.number_input("Δt [s]", value=0.002, min_value=1e-5, step=0.001, format="%.6f", key=f"{prefix}_dt")

    with st.expander("Animation / performance", expanded=False):
        fps_anim = st.slider("Animation FPS", 10, 60, 30, 5, key=f"{prefix}_fps_anim")
        max_frames = st.slider("Max frames", 120, 900, 420, 20, key=f"{prefix}_max_frames")

        trail_on = st.checkbox("Show rotor tip trail", value=False, key=f"{prefix}_trail_on")
        trail_max_points = st.slider("Trail points", 20, 600, 140, 10, key=f"{prefix}_trail_max_points")

        max_plot_pts = st.slider("Plot points (downsample)", 400, 12000, 2600, 200, key=f"{prefix}_max_plot_pts")

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
        fps_anim=int(fps_anim),
        max_frames=int(max_frames),
        trail_on=bool(trail_on),
        trail_max_points=int(trail_max_points),
        max_plot_pts=int(max_plot_pts),
    )


def run(controls: Controls) -> Tuple[Cfg, Out]:
    params = {
        "R": controls["R"],
        "L": controls["L"],
        "Ke": controls["Ke"],
        "Kt": controls["Kt"],
        "J": controls["J"],
        "bm": controls["bm"],
        "v_mode": controls["v_mode"],
        "V0": controls["V0"],
        "v_offset": controls["v_offset"],
        "t_step": controls["t_step"],
        "v_freq": controls["v_freq"],
        "v_duty": controls["v_duty"],
        "load_mode": controls["load_mode"],
        "tau_load": controls["tau_load"],
        "b_load": controls["b_load"],
        "tau_c": controls["tau_c"],
        "omega_eps": controls["omega_eps"],
    }
    ic = {"i0": controls["i0"], "omega0": controls["omega0"]}
    return run_dc_motor(params, ic, controls["t0"], controls["t1"], controls["dt"])


def caption(cfg: Cfg, out: Out) -> str:
    T = np.asarray(out.get("T", []), dtype=float)
    dt_eff = float(np.mean(np.diff(T))) if len(T) > 1 else float(cfg.get("dt", 0.002))
    return f"Δt ≈ {dt_eff:.6f} s · N = {len(T)}"


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
