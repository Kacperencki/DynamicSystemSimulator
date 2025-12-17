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
from apps.streamlit.runners.inverted_runner import run_ip_open, run_ip_closed
from apps.streamlit.components.dashboards.inverted_pendulum_dashboard import make_inverted_pendulum_dashboard

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    "preset",
    "ctrl_mode",
    "mode",
    "mass_model",
    "length", "mass", "cart_mass", "g",
    "b_cart", "coulomb_cart", "b_pend", "coulomb_pend", "coulomb_k",
    "cart_drive_amp", "cart_drive_freq", "cart_drive_phase",
    "pend_drive_amp", "pend_drive_freq", "pend_drive_phase",
    "x0", "xdot0", "th0", "thdot0",
    # LQR
    "q_x", "q_xdot", "q_theta", "q_thetad", "u_max",
    # Swing
    "k_e", "su_u_max",
    # Switch
    "engage_angle_deg", "engage_speed_rad_s", "engage_cart_speed",
    "dropout_angle_deg", "dropout_speed_rad_s", "dropout_cart_speed",
    "allow_dropout", "blend_time", "du_max",
    # sim
    "t0", "t1", "dt",
    "solver_method", "rtol", "atol",
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default (open-loop)": dict(
        ctrl_mode="Open-loop (no control)",
        mode="ideal",
        mass_model="point",
        length=1.0, mass=0.2, cart_mass=0.5, g=9.81,
        b_cart=0.0, coulomb_cart=0.0, b_pend=0.0, coulomb_pend=0.0, coulomb_k=1e3,
        cart_drive_amp=0.0, cart_drive_freq=0.0, cart_drive_phase=0.0,
        pend_drive_amp=0.0, pend_drive_freq=0.0, pend_drive_phase=0.0,
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(10.0)), thdot0=0.0,
        q_x=1.0, q_xdot=1.0, q_theta=60.0, q_thetad=1.0, u_max=20.0,
        k_e=None, su_u_max=None,
        engage_angle_deg=25.0, engage_speed_rad_s=9.0, engage_cart_speed=6.0,
        dropout_angle_deg=45.0, dropout_speed_rad_s=30.0, dropout_cart_speed=10.0,
        allow_dropout=True, blend_time=0.12, du_max=800.0,
        t0=0.0, t1=10.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=500, max_plot_pts=3000,
        trail_on=False, trail_max_points=220,
    ),
    "LQR stabilization": dict(
        ctrl_mode="LQR stabilizer",
        mode="ideal",
        mass_model="point",
        length=1.0, mass=0.2, cart_mass=0.5, g=9.81,
        b_cart=0.0, coulomb_cart=0.0, b_pend=0.0, coulomb_pend=0.0, coulomb_k=1e3,
        cart_drive_amp=0.0, cart_drive_freq=0.0, cart_drive_phase=0.0,
        pend_drive_amp=0.0, pend_drive_freq=0.0, pend_drive_phase=0.0,
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(6.0)), thdot0=0.0,
        q_x=1.0, q_xdot=2.0, q_theta=120.0, q_thetad=2.0, u_max=20.0,
        k_e=None, su_u_max=None,
        engage_angle_deg=25.0, engage_speed_rad_s=9.0, engage_cart_speed=6.0,
        dropout_angle_deg=45.0, dropout_speed_rad_s=30.0, dropout_cart_speed=10.0,
        allow_dropout=True, blend_time=0.12, du_max=800.0,
        t0=0.0, t1=8.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=450, max_plot_pts=2500,
        trail_on=False, trail_max_points=220,
    ),
    "Swing-up + LQR": dict(
        ctrl_mode="Swing-up + LQR (simple)",
        mode="ideal",
        mass_model="point",
        length=1.0, mass=0.2, cart_mass=0.5, g=9.81,
        b_cart=0.0, coulomb_cart=0.0, b_pend=0.0, coulomb_pend=0.0, coulomb_k=1e3,
        cart_drive_amp=0.0, cart_drive_freq=0.0, cart_drive_phase=0.0,
        pend_drive_amp=0.0, pend_drive_freq=0.0, pend_drive_phase=0.0,
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(170.0)), thdot0=0.0,
        q_x=1.0, q_xdot=2.0, q_theta=150.0, q_thetad=3.0, u_max=25.0,
        k_e=None, su_u_max=None,
        engage_angle_deg=25.0, engage_speed_rad_s=9.0, engage_cart_speed=6.0,
        dropout_angle_deg=45.0, dropout_speed_rad_s=30.0, dropout_cart_speed=10.0,
        allow_dropout=True, blend_time=0.12, du_max=800.0,
        t0=0.0, t1=14.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=700, max_plot_pts=5000,
        trail_on=True, trail_max_points=260,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Default (open-loop)")

    ctrl_mode = st.selectbox(
        "Control mode",
        ["Open-loop (no control)", "LQR stabilizer", "Swing-up only", "Swing-up + LQR (simple)"],
        index=0,
        key=f"{prefix}_ctrl_mode",
    )

    mode = st.selectbox(
        "Plant mode",
        ["ideal", "damped", "driven"],
        index=0,
        key=f"{prefix}_mode",
    )

    mass_model = st.selectbox(
        "Mass model",
        ["point", "uniform"],
        index=0,
        key=f"{prefix}_mass_model",
    )

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Clear")

        with st.expander("Physical parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                length = st.number_input("l [m]", value=1.0, min_value=0.0, key=f"{prefix}_length")
                mass = st.number_input("m [kg]", value=0.2, min_value=0.0, key=f"{prefix}_mass")
            with c2:
                cart_mass = st.number_input("M [kg]", value=0.5, min_value=0.0, key=f"{prefix}_cart_mass")
                g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g")
            with c3:
                # keep as an explicit control because it can be useful for numerical smoothing
                coulomb_k = st.number_input("Coulomb smoothing k", value=1e3, min_value=1.0, format="%.3g", key=f"{prefix}_coulomb_k")

        # friction defaults (persist between modes)
        b_cart = float(st.session_state.get(f"{prefix}_b_cart", 0.0))
        coulomb_cart = float(st.session_state.get(f"{prefix}_coulomb_cart", 0.0))
        b_pend = float(st.session_state.get(f"{prefix}_b_pend", 0.0))
        coulomb_pend = float(st.session_state.get(f"{prefix}_coulomb_pend", 0.0))

        # drive defaults (persist between modes)
        cart_drive_amp = float(st.session_state.get(f"{prefix}_cart_drive_amp", 0.0))
        cart_drive_freq = float(st.session_state.get(f"{prefix}_cart_drive_freq", 0.0))
        cart_drive_phase = float(st.session_state.get(f"{prefix}_cart_drive_phase", 0.0))
        pend_drive_amp = float(st.session_state.get(f"{prefix}_pend_drive_amp", 0.0))
        pend_drive_freq = float(st.session_state.get(f"{prefix}_pend_drive_freq", 0.0))
        pend_drive_phase = float(st.session_state.get(f"{prefix}_pend_drive_phase", 0.0))

        if mode in ("damped", "driven"):
            with st.expander("Friction", expanded=False):
                f1, f2 = st.columns(2)
                with f1:
                    b_cart = st.number_input("b_cart", value=b_cart, min_value=0.0, key=f"{prefix}_b_cart")
                    coulomb_cart = st.number_input("F_c (cart)", value=coulomb_cart, min_value=0.0, key=f"{prefix}_coulomb_cart")
                with f2:
                    b_pend = st.number_input("b_pend", value=b_pend, min_value=0.0, key=f"{prefix}_b_pend")
                    coulomb_pend = st.number_input("F_c (pend)", value=coulomb_pend, min_value=0.0, key=f"{prefix}_coulomb_pend")
        else:
            b_cart = coulomb_cart = b_pend = coulomb_pend = 0.0

        if mode == "driven":
            with st.expander("External drives", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    cart_drive_amp = st.number_input("Cart drive A", value=cart_drive_amp, min_value=0.0, key=f"{prefix}_cart_drive_amp")
                    cart_drive_freq = st.number_input("Cart drive ω", value=cart_drive_freq, min_value=0.0, key=f"{prefix}_cart_drive_freq")
                    cart_drive_phase = st.number_input("Cart drive φ", value=cart_drive_phase, key=f"{prefix}_cart_drive_phase")
                with d2:
                    pend_drive_amp = st.number_input("Pend drive A", value=pend_drive_amp, min_value=0.0, key=f"{prefix}_pend_drive_amp")
                    pend_drive_freq = st.number_input("Pend drive ω", value=pend_drive_freq, min_value=0.0, key=f"{prefix}_pend_drive_freq")
                    pend_drive_phase = st.number_input("Pend drive φ", value=pend_drive_phase, key=f"{prefix}_pend_drive_phase")
        else:
            cart_drive_amp = cart_drive_freq = cart_drive_phase = 0.0
            pend_drive_amp = pend_drive_freq = pend_drive_phase = 0.0

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.number_input("x(0) [m]", value=0.0, format="%.6g", key=f"{prefix}_x0")
                xdot0 = st.number_input("ẋ(0) [m/s]", value=0.0, format="%.6g", key=f"{prefix}_xdot0")
            with c2:
                th0 = st.number_input("θ(0) [rad]", value=float(np.deg2rad(10.0)), format="%.6g", key=f"{prefix}_th0")
                thdot0 = st.number_input("θ̇(0) [rad/s]", value=0.0, format="%.6g", key=f"{prefix}_thdot0")

        # controller settings
        lqr_settings: Dict[str, Any] = {}
        swing_settings: Dict[str, Any] = {}
        switch_settings: Dict[str, Any] = {}

        if ctrl_mode == "LQR stabilizer":
            with st.expander("LQR settings", expanded=False):
                q1, q2 = st.columns(2)
                with q1:
                    q_x = st.number_input("q_x", value=1.0, min_value=0.0, key=f"{prefix}_q_x")
                    q_xdot = st.number_input("q_xdot", value=1.0, min_value=0.0, key=f"{prefix}_q_xdot")
                with q2:
                    q_theta = st.number_input("q_theta", value=60.0, min_value=0.0, key=f"{prefix}_q_theta")
                    q_thetad = st.number_input("q_thetad", value=1.0, min_value=0.0, key=f"{prefix}_q_thetad")
                u_max = st.number_input("u_max [N]", value=20.0, min_value=0.0, key=f"{prefix}_u_max")
                lqr_settings = dict(q_x=q_x, q_xdot=q_xdot, q_theta=q_theta, q_thetad=q_thetad, u_max=u_max)

        elif ctrl_mode == "Swing-up only":
            with st.expander("Swing-up settings", expanded=False):
                k_e = st.number_input("k_e (energy gain)", value=st.session_state.get(f"{prefix}_k_e", 1.0) or 1.0, min_value=0.0, key=f"{prefix}_k_e")
                su_u_max = st.number_input("u_max [N] (optional)", value=st.session_state.get(f"{prefix}_su_u_max", 25.0) or 25.0, min_value=0.0, key=f"{prefix}_su_u_max")
                swing_settings = dict(k_e=k_e, u_max=su_u_max)

        elif ctrl_mode == "Swing-up + LQR (simple)":
            with st.expander("Swing-up settings", expanded=False):
                k_e = st.number_input("k_e (energy gain)", value=st.session_state.get(f"{prefix}_k_e", 1.0) or 1.0, min_value=0.0, key=f"{prefix}_k_e")
                su_u_max = st.number_input("u_max [N]", value=st.session_state.get(f"{prefix}_su_u_max", 25.0) or 25.0, min_value=0.0, key=f"{prefix}_su_u_max")
                swing_settings = dict(k_e=k_e, u_max=su_u_max)

            with st.expander("LQR settings", expanded=False):
                q1, q2 = st.columns(2)
                with q1:
                    q_x = st.number_input("q_x", value=1.0, min_value=0.0, key=f"{prefix}_q_x")
                    q_xdot = st.number_input("q_xdot", value=1.0, min_value=0.0, key=f"{prefix}_q_xdot")
                with q2:
                    q_theta = st.number_input("q_theta", value=60.0, min_value=0.0, key=f"{prefix}_q_theta")
                    q_thetad = st.number_input("q_thetad", value=1.0, min_value=0.0, key=f"{prefix}_q_thetad")
                u_max = st.number_input("u_max [N]", value=20.0, min_value=0.0, key=f"{prefix}_u_max")
                lqr_settings = dict(q_x=q_x, q_xdot=q_xdot, q_theta=q_theta, q_thetad=q_thetad, u_max=u_max)

            with st.expander("Switcher settings", expanded=False):
                s1, s2, s3 = st.columns(3)
                with s1:
                    engage_angle_deg = st.number_input("Engage angle [deg]", value=25.0, min_value=0.0, key=f"{prefix}_engage_angle_deg")
                    engage_speed_rad_s = st.number_input("Engage θ̇ [rad/s]", value=9.0, min_value=0.0, key=f"{prefix}_engage_speed_rad_s")
                with s2:
                    engage_cart_speed = st.number_input("Engage ẋ [m/s]", value=6.0, min_value=0.0, key=f"{prefix}_engage_cart_speed")
                    dropout_angle_deg = st.number_input("Dropout angle [deg]", value=45.0, min_value=0.0, key=f"{prefix}_dropout_angle_deg")
                with s3:
                    dropout_speed_rad_s = st.number_input("Dropout θ̇ [rad/s]", value=30.0, min_value=0.0, key=f"{prefix}_dropout_speed_rad_s")
                    dropout_cart_speed = st.number_input("Dropout ẋ [m/s]", value=10.0, min_value=0.0, key=f"{prefix}_dropout_cart_speed")
                allow_dropout = st.checkbox("Allow dropout", value=True, key=f"{prefix}_allow_dropout")
                blend_time = st.number_input("Blend time [s] (if supported)", value=0.12, min_value=0.0, key=f"{prefix}_blend_time")
                du_max = st.number_input("Δu max [N/s] (if supported)", value=800.0, min_value=0.0, key=f"{prefix}_du_max")
                switch_settings = dict(
                    engage_angle_deg=engage_angle_deg,
                    engage_speed_rad_s=engage_speed_rad_s,
                    engage_cart_speed=engage_cart_speed,
                    dropout_angle_deg=dropout_angle_deg,
                    dropout_speed_rad_s=dropout_speed_rad_s,
                    dropout_cart_speed=dropout_cart_speed,
                    allow_dropout=allow_dropout,
                    blend_time=blend_time,
                    du_max=du_max,
                )

        t0, t1, dt = simulation_time(prefix, expanded=False, t0_default=0.0, t1_default=10.0, dt_default=0.01, dt_min=1e-5, dt_step=0.001, dt_format="%.6f")

        solver = solver_settings(prefix, expanded=False)

        perf = animation_performance(
            prefix,
            title="Animation / performance",
            expanded=False,
            layout="single",
            fps=SliderSpec("Animation FPS", 10, 60, 30, 5),
            max_frames=SliderSpec("Max frames", 200, 1200, 600, 20),
            max_plot_pts=SliderSpec("Plot points (downsample)", 800, 20000, 6000, 200),
            trail_default=False,
            trail_checkbox_label="Show cart trace (x)",
            trail_max_points=SliderSpec("Trace max points", 50, 700, 260, 10),
        )

        save_run, log_dir, run_name = logging_settings(prefix, expanded=False)

    return dict(
        run_clicked=run_clicked,
        ctrl_mode=str(ctrl_mode),
        mode=str(mode),
        mass_model=str(mass_model),
        length=float(length),
        mass=float(mass),
        cart_mass=float(cart_mass),
        g=float(g),
        b_cart=float(b_cart),
        coulomb_cart=float(coulomb_cart),
        b_pend=float(b_pend),
        coulomb_pend=float(coulomb_pend),
        coulomb_k=float(coulomb_k),
        cart_drive_amp=float(cart_drive_amp),
        cart_drive_freq=float(cart_drive_freq),
        cart_drive_phase=float(cart_drive_phase),
        pend_drive_amp=float(pend_drive_amp),
        pend_drive_freq=float(pend_drive_freq),
        pend_drive_phase=float(pend_drive_phase),
        x0=float(x0),
        xdot0=float(xdot0),
        th0=float(th0),
        thdot0=float(thdot0),
        lqr_settings=lqr_settings,
        swing_settings=swing_settings,
        switch_settings=switch_settings,
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
        length=controls["length"],
        mass=controls["mass"],
        cart_mass=controls["cart_mass"],
        g=controls["g"],
        mass_model=controls["mass_model"],
        b_cart=controls["b_cart"],
        coulomb_cart=controls["coulomb_cart"],
        b_pend=controls["b_pend"],
        coulomb_pend=controls["coulomb_pend"],
        coulomb_k=controls["coulomb_k"],
        cart_drive_amp=controls["cart_drive_amp"],
        cart_drive_freq=controls["cart_drive_freq"],
        cart_drive_phase=controls["cart_drive_phase"],
        pend_drive_amp=controls["pend_drive_amp"],
        pend_drive_freq=controls["pend_drive_freq"],
        pend_drive_phase=controls["pend_drive_phase"],
    )

    ic = dict(
        x0=controls["x0"],
        xdot0=controls["xdot0"],
        th0=controls["th0"],
        thdot0=controls["thdot0"],
    )

    if controls["ctrl_mode"] == "Open-loop (no control)":
        return run_ip_open(
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

    return run_ip_closed(
        ctrl_mode=controls["ctrl_mode"],
        params=params,
        lqr_set=controls.get("lqr_settings", {}),
        swing_set=controls.get("swing_settings", {}),
        switch_set=controls.get("switch_settings", {}),
        ic=ic,
        t0=controls["t0"],
        t1=controls["t1"],
        dt=controls["dt"],
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
    return make_inverted_pendulum_dashboard(cfg, out, ui)


def get_spec() -> SystemSpec:
    return SystemSpec(
        id="ip",
        title="Inverted pendulum / cart–pole",
        controls=controls,
        run=run,
        caption=caption,
        make_dashboard=make_dashboard,
    )
