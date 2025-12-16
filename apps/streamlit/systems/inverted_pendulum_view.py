from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import streamlit as st

from apps.streamlit.layout import SystemSpec
from apps.streamlit.runners.inverted_runner import run_ip_open, run_ip_closed
from apps.streamlit.components.controls_common import reset_defaults_button
from apps.streamlit.components.dashboards.inverted_pendulum_dashboard import (
    make_inverted_pendulum_dashboard,
)

Cfg = Dict[str, Any]
Out = Dict[str, Any]
Controls = Dict[str, Any]


RESET_KEYS = [
    # main selections
    "mode", "mass_model", "ctrl_mode",
    # physical
    "length", "mass", "cart_mass", "g",
    # friction
    "b_cart", "coulomb_cart", "b_pend", "coulomb_pend", "coulomb_k",
    # harmonic drives
    "cart_drive_amp", "cart_drive_freq", "cart_drive_phase",
    "pend_drive_amp", "pend_drive_freq", "pend_drive_phase",
    # initial conditions
    "x0", "xdot0", "th0", "thdot0",
    # sim
    "t0", "t1", "dt",
    # lqr
    "q_theta", "q_x", "u_max_lqr",
    # swing
    "k_e", "u_max_swing",
    # switcher
    "engage_angle", "engage_speed", "engage_cart_speed",
    "dropout_angle", "dropout_speed", "dropout_cart_speed", "allow_dropout",
    # perf / viz
    "fps_anim", "max_frames", "trail_on", "trail_max_points", "max_plot_pts",
]


def controls(prefix: str) -> Controls:
    c1, c2 = st.columns([1, 1])
    with c1:
        run_clicked = st.button("Run", key=f"{prefix}_run", type="primary", width='stretch')
    with c2:
        reset_defaults_button(prefix, RESET_KEYS, label="Reset")

    ctrl_mode = st.selectbox(
        "Control mode",
        [
            "Open-loop (no control)",
            "LQR stabilizer",
            "Swing-up only",
            "Swing-up + LQR (simple)",
        ],
        index=0,
        key=f"{prefix}_ctrl_mode",
    )

    mode = st.selectbox(
        "Plant mode",
        ["ideal", "damped_cart", "damped_pend", "damped_both", "driven", "dc_driven"],
        index=0,
        key=f"{prefix}_mode",
    )

    with st.expander("Physical parameters", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            length = st.number_input("Pole length l [m]", value=0.30, min_value=0.0, key=f"{prefix}_length")
        with r2:
            mass = st.number_input("Pole mass m [kg]", value=0.20, min_value=0.0, key=f"{prefix}_mass")
        with r3:
            cart_mass = st.number_input("Cart mass M [kg]", value=0.50, min_value=0.0, key=f"{prefix}_cart_mass")
        g = st.number_input("Gravity g [m/s²]", value=9.81, key=f"{prefix}_g")
        mass_model = st.selectbox("Mass model", ["point", "uniform"], index=0, key=f"{prefix}_mass_model")

    # defaults (so switching mode doesn't hard-reset)
    b_cart = float(st.session_state.get(f"{prefix}_b_cart", 0.0))
    coulomb_cart = float(st.session_state.get(f"{prefix}_coulomb_cart", 0.0))
    b_pend = float(st.session_state.get(f"{prefix}_b_pend", 0.0))
    coulomb_pend = float(st.session_state.get(f"{prefix}_coulomb_pend", 0.0))
    coulomb_k = float(st.session_state.get(f"{prefix}_coulomb_k", 1000.0))

    cart_drive_amp = float(st.session_state.get(f"{prefix}_cart_drive_amp", 0.0))
    cart_drive_freq = float(st.session_state.get(f"{prefix}_cart_drive_freq", 2.0))
    cart_drive_phase = float(st.session_state.get(f"{prefix}_cart_drive_phase", 0.0))

    pend_drive_amp = float(st.session_state.get(f"{prefix}_pend_drive_amp", 0.0))
    pend_drive_freq = float(st.session_state.get(f"{prefix}_pend_drive_freq", 2.0))
    pend_drive_phase = float(st.session_state.get(f"{prefix}_pend_drive_phase", 0.0))

    if mode == "ideal":
        b_cart = coulomb_cart = b_pend = coulomb_pend = 0.0
        cart_drive_amp = cart_drive_freq = cart_drive_phase = 0.0
        pend_drive_amp = pend_drive_freq = pend_drive_phase = 0.0

    elif mode in ("damped_cart", "damped_pend", "damped_both"):
        with st.expander("Friction (viscous + Coulomb)", expanded=False):
            if mode in ("damped_cart", "damped_both"):
                f1, f2 = st.columns(2)
                with f1:
                    b_cart = st.number_input("Cart viscous b_c [N·s/m]", value=0.0, min_value=0.0, key=f"{prefix}_b_cart")
                with f2:
                    coulomb_cart = st.number_input("Cart Coulomb F_c [N]", value=0.0, min_value=0.0, key=f"{prefix}_coulomb_cart")
            else:
                b_cart = 0.0
                coulomb_cart = 0.0

            if mode in ("damped_pend", "damped_both"):
                f3, f4 = st.columns(2)
                with f3:
                    b_pend = st.number_input("Pend viscous b_p [N·m·s]", value=0.0, min_value=0.0, key=f"{prefix}_b_pend")
                with f4:
                    coulomb_pend = st.number_input("Pend Coulomb τ_c [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_coulomb_pend")
            else:
                b_pend = 0.0
                coulomb_pend = 0.0

            coulomb_k = st.number_input("Coulomb smoothing k", value=1000.0, min_value=1.0, key=f"{prefix}_coulomb_k")

        cart_drive_amp = cart_drive_freq = cart_drive_phase = 0.0
        pend_drive_amp = pend_drive_freq = pend_drive_phase = 0.0

    elif mode in ("driven", "dc_driven"):
        with st.expander("Friction (viscous + Coulomb)", expanded=False):
            f1, f2 = st.columns(2)
            with f1:
                b_cart = st.number_input("Cart viscous b_c [N·s/m]", value=0.0, min_value=0.0, key=f"{prefix}_b_cart")
            with f2:
                coulomb_cart = st.number_input("Cart Coulomb F_c [N]", value=0.0, min_value=0.0, key=f"{prefix}_coulomb_cart")

            f3, f4 = st.columns(2)
            with f3:
                b_pend = st.number_input("Pend viscous b_p [N·m·s]", value=0.0, min_value=0.0, key=f"{prefix}_b_pend")
            with f4:
                coulomb_pend = st.number_input("Pend Coulomb τ_c [N·m]", value=0.0, min_value=0.0, key=f"{prefix}_coulomb_pend")

            coulomb_k = st.number_input("Coulomb smoothing k", value=1000.0, min_value=1.0, key=f"{prefix}_coulomb_k")

        title = "Harmonic drives" if mode == "driven" else "DC-like drives (set ω=0 and phase≈π/2 for constant)"
        with st.expander(title, expanded=False):
            st.markdown("**Cart force drive**")
            p1, p2, p3 = st.columns(3)
            with p1:
                cart_drive_amp = st.number_input("A_F [N]", value=0.0, key=f"{prefix}_cart_drive_amp")
            with p2:
                cart_drive_freq = st.number_input("ω_F [rad/s]", value=2.0, min_value=0.0, key=f"{prefix}_cart_drive_freq")
            with p3:
                cart_drive_phase = st.number_input("φ_F [rad]", value=0.0, key=f"{prefix}_cart_drive_phase")

            st.markdown("**Pivot torque drive**")
            q1, q2, q3 = st.columns(3)
            with q1:
                pend_drive_amp = st.number_input("A_τ [N·m]", value=0.0, key=f"{prefix}_pend_drive_amp")
            with q2:
                pend_drive_freq = st.number_input("ω_τ [rad/s]", value=2.0, min_value=0.0, key=f"{prefix}_pend_drive_freq")
            with q3:
                pend_drive_phase = st.number_input("φ_τ [rad]", value=0.0, key=f"{prefix}_pend_drive_phase")

    with st.expander("Initial conditions", expanded=False):
        x0 = st.number_input("x(0) [m]", value=0.0, key=f"{prefix}_x0")
        xdot0 = st.number_input("ẋ(0) [m/s]", value=0.0, key=f"{prefix}_xdot0")
        th0 = st.number_input("θ(0) [rad]", value=0.05, key=f"{prefix}_th0")
        thdot0 = st.number_input("θ̇(0) [rad/s]", value=0.0, key=f"{prefix}_thdot0")

    # controller settings
    lqr_settings: Dict[str, Any] = {}
    swing_settings: Dict[str, Any] = {}
    switch_settings: Dict[str, Any] = {}

    if ctrl_mode in ("LQR stabilizer", "Swing-up + LQR (simple)"):
        with st.expander("LQR settings", expanded=False):
            q_theta = st.slider("Angle weight q_θ", 10.0, 200.0, 50.0, 5.0, key=f"{prefix}_q_theta")
            q_x = st.slider("Cart position weight q_x", 0.1, 10.0, 1.0, 0.1, key=f"{prefix}_q_x")
            u_max_lqr = st.slider("Max control force |F| [N]", 5.0, 50.0, 20.0, 1.0, key=f"{prefix}_u_max_lqr")

        lqr_settings = dict(
            q_x=float(q_x),
            q_xdot=0.5,
            q_theta=float(q_theta),
            q_thetad=8.0,
            r_u=0.1,
            u_max=float(u_max_lqr),
        )

    if ctrl_mode in ("Swing-up only", "Swing-up + LQR (simple)"):
        with st.expander("Swing-up settings", expanded=False):
            k_e = st.slider("Energy gain k_e", 0.1, 50.0, 10.0, 0.5, key=f"{prefix}_k_e")
            u_max_swing = st.slider("Max swing-up force |F| [N]", 5.0, 80.0, 40.0, 1.0, key=f"{prefix}_u_max_swing")
        swing_settings = dict(k_e=float(k_e), u_max=float(u_max_swing))

    if ctrl_mode == "Swing-up + LQR (simple)":
        with st.expander("Switcher (Swing-up ↔ LQR)", expanded=False):
            engage_angle = st.slider("Engage LQR below |θ| [deg]", 5.0, 30.0, 12.0, 1.0, key=f"{prefix}_engage_angle")
            engage_speed = st.slider("Engage LQR below |θ̇| [rad/s]", 0.5, 10.0, 1.5, 0.5, key=f"{prefix}_engage_speed")
            engage_cart_speed = st.slider("Engage LQR below |ẋ| [m/s]", 0.5, 10.0, 6.0, 0.5, key=f"{prefix}_engage_cart_speed")
            dropout_angle = st.slider("Dropout above |θ| [deg]", 5.0, 60.0, 25.0, 1.0, key=f"{prefix}_dropout_angle")
            dropout_speed = st.slider("Dropout above |θ̇| [rad/s]", 2.0, 40.0, 20.0, 1.0, key=f"{prefix}_dropout_speed")
            dropout_cart_speed = st.slider("Dropout above |ẋ| [m/s]", 1.0, 20.0, 10.0, 1.0, key=f"{prefix}_dropout_cart_speed")
            allow_dropout = st.checkbox("Allow dropout", value=True, key=f"{prefix}_allow_dropout")
        switch_settings = dict(
            engage_angle_deg=float(engage_angle),
            engage_speed_rad_s=float(engage_speed),
            engage_cart_speed=float(engage_cart_speed),
            dropout_angle_deg=float(dropout_angle),
            dropout_speed_rad_s=float(dropout_speed),
            dropout_cart_speed=float(dropout_cart_speed),
            allow_dropout=bool(allow_dropout),
        )

    with st.expander("Simulation time", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            t0 = st.number_input("t₀ [s]", value=0.0, key=f"{prefix}_t0")
        with c2:
            t1 = st.number_input("t₁ [s]", value=10.0, key=f"{prefix}_t1")
        with c3:
            dt = st.number_input("Δt [s]", value=0.01, min_value=1e-4, step=0.001, format="%.4f", key=f"{prefix}_dt")

    with st.expander("Performance / visualization", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            fps_anim = st.slider("Animation FPS", 5, 60, 30, 1, key=f"{prefix}_fps_anim")
            max_frames = st.slider("Max animation frames", 60, 1200, 360, 10, key=f"{prefix}_max_frames")
            max_plot_pts = st.slider("Max plot points", 500, 10000, 2000, 100, key=f"{prefix}_max_plot_pts")
        with p2:
            trail_on = st.checkbox("Show tip trail", value=False, key=f"{prefix}_trail_on")
            trail_max_points = st.slider("Trail points", 30, 600, 180, 10, key=f"{prefix}_trail_max_points")

    return dict(
        run_clicked=run_clicked,
        ctrl_mode=ctrl_mode,
        mode=mode,
        mass_model=mass_model,
        length=length,
        mass=mass,
        cart_mass=cart_mass,
        g=g,
        b_cart=b_cart,
        coulomb_cart=coulomb_cart,
        b_pend=b_pend,
        coulomb_pend=coulomb_pend,
        coulomb_k=coulomb_k,
        cart_drive_amp=cart_drive_amp,
        cart_drive_freq=cart_drive_freq,
        cart_drive_phase=cart_drive_phase,
        pend_drive_amp=pend_drive_amp,
        pend_drive_freq=pend_drive_freq,
        pend_drive_phase=pend_drive_phase,
        x0=x0,
        xdot0=xdot0,
        th0=th0,
        thdot0=thdot0,
        lqr_settings=lqr_settings,
        swing_settings=swing_settings,
        switch_settings=switch_settings,
        t0=t0,
        t1=t1,
        dt=dt,
        fps_anim=fps_anim,
        max_frames=max_frames,
        max_plot_pts=max_plot_pts,
        trail_on=trail_on,
        trail_max_points=trail_max_points,
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
        return run_ip_open(params, ic, controls["t0"], controls["t1"], controls["dt"])

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
