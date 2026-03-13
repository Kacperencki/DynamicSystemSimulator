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


def _lqr_settings_expander(prefix: str) -> Dict[str, Any]:
    """Render the LQR weights expander and return the settings dict."""
    with st.expander("LQR settings", expanded=False):
        q1, q2 = st.columns(2)
        with q1:
            q_x = st.number_input("x weight", value=1.0, min_value=0.0, key=f"{prefix}_q_x",
                                  help="State cost weight for cart position x. Higher = penalise drift more.")
            q_xdot = st.number_input("ẋ weight", value=1.0, min_value=0.0, key=f"{prefix}_q_xdot",
                                     help="State cost weight for cart velocity.")
        with q2:
            q_theta = st.number_input("θ weight", value=60.0, min_value=0.0, key=f"{prefix}_q_theta",
                                      help="State cost weight for pole angle. Higher = more aggressive angle correction.")
            q_thetad = st.number_input("θ̇ weight", value=1.0, min_value=0.0, key=f"{prefix}_q_thetad",
                                       help="State cost weight for pole angular velocity.")
        u_max = st.number_input("Force limit [N]", value=20.0, min_value=0.0, key=f"{prefix}_u_max",
                                help="Maximum cart force the controller can apply (actuator saturation).")
    return dict(q_x=q_x, q_xdot=q_xdot, q_theta=q_theta, q_thetad=q_thetad, u_max=u_max)


def _swingup_settings_expander(prefix: str) -> Dict[str, Any]:
    """Render the Swing-up settings expander and return the settings dict."""
    with st.expander("Swing-up settings", expanded=False):
        k_e = st.number_input(
            "Energy gain (0 = auto)",
            value=float(st.session_state.get(f"{prefix}_k_e") or 0.0),
            min_value=0.0, key=f"{prefix}_k_e",
            help="Energy pumping gain. 0 = use physics-based default (5 / (m·lc)).",
        )
        su_u_max = st.number_input(
            "Max force [N]",
            value=float(st.session_state.get(f"{prefix}_su_u_max") or 25.0),
            min_value=0.1, key=f"{prefix}_su_u_max",
            help="Maximum cart force the swing-up controller can apply.",
        )
    return dict(k_e=k_e, u_max=su_u_max)


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
    "Open-loop (free fall)": dict(
        ctrl_mode="Open-loop (no control)",
        mode="ideal",
        mass_model="point",
        length=1.0, mass=0.2, cart_mass=0.5, g=9.81,
        b_cart=0.0, coulomb_cart=0.0, b_pend=0.0, coulomb_pend=0.0, coulomb_k=1e3,
        cart_drive_amp=0.0, cart_drive_freq=0.0, cart_drive_phase=0.0,
        pend_drive_amp=0.0, pend_drive_freq=0.0, pend_drive_phase=0.0,
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(10.0)), thdot0=0.0,
        q_x=1.0, q_xdot=1.0, q_theta=60.0, q_thetad=1.0, u_max=20.0,
        k_e=0.0, su_u_max=25.0,
        engage_angle_deg=25.0, engage_speed_rad_s=9.0, engage_cart_speed=6.0,
        dropout_angle_deg=45.0, dropout_speed_rad_s=30.0, dropout_cart_speed=10.0,
        allow_dropout=True, blend_time=0.12, du_max=800.0,
        t0=0.0, t1=6.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=400, max_plot_pts=2000,
        trail_on=False, trail_max_points=220,
    ),
    "LQR stabilisation": dict(
        ctrl_mode="LQR stabilizer",
        mode="ideal",
        mass_model="point",
        length=1.0, mass=0.2, cart_mass=0.5, g=9.81,
        b_cart=0.0, coulomb_cart=0.0, b_pend=0.0, coulomb_pend=0.0, coulomb_k=1e3,
        cart_drive_amp=0.0, cart_drive_freq=0.0, cart_drive_phase=0.0,
        pend_drive_amp=0.0, pend_drive_freq=0.0, pend_drive_phase=0.0,
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(15.0)), thdot0=0.0,
        q_x=1.0, q_xdot=2.0, q_theta=120.0, q_thetad=2.0, u_max=20.0,
        k_e=0.0, su_u_max=25.0,
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
        x0=0.0, xdot0=0.0, th0=float(np.deg2rad(175.0)), thdot0=0.0,
        q_x=1.0, q_xdot=2.0, q_theta=150.0, q_thetad=3.0, u_max=25.0,
        k_e=0.0, su_u_max=25.0,
        engage_angle_deg=25.0, engage_speed_rad_s=9.0, engage_cart_speed=6.0,
        dropout_angle_deg=45.0, dropout_speed_rad_s=30.0, dropout_cart_speed=10.0,
        allow_dropout=True, blend_time=0.12, du_max=800.0,
        t0=0.0, t1=15.0, dt=0.01,
        solver_method="RK45", rtol=1e-4, atol=1e-6,
        fps_anim=30, max_frames=700, max_plot_pts=5000,
        trail_on=True, trail_max_points=300,
    ),
}


def controls(prefix: str) -> Controls:
    presets_selector(prefix, PRESETS, label="Preset", default_name="Open-loop (free fall)")

    ctrl_mode = st.selectbox(
        "Control mode",
        ["Open-loop (no control)", "LQR stabilizer", "Swing-up only", "Swing-up + LQR (simple)"],
        index=0,
        key=f"{prefix}_ctrl_mode",
        help="Open-loop: no control applied; LQR: linear-quadratic regulator near upright; Swing-up: energy-based controller to raise the pole; Swing-up + LQR: automatic handoff from swing-up to LQR.",
    )

    mode = st.selectbox(
        "Plant mode",
        ["ideal", "damped", "driven"],
        index=0,
        key=f"{prefix}_mode",
        help="ideal: no friction/drive; damped: adds viscous + Coulomb friction; driven: adds external sinusoidal forces.",
    )

    mass_model = st.selectbox(
        "Mass model",
        ["point", "uniform"],
        index=0,
        key=f"{prefix}_mass_model",
        help="point: all pole mass at tip; uniform: mass distributed evenly along the pole (moment of inertia J = mL²/3).",
    )

    with st.form(key=f"{prefix}_form"):
        run_clicked = run_clear_row_form(prefix, RESET_KEYS, clear_label="Reset", default_preset=PRESETS.get("Open-loop (free fall)", {}), default_preset_name="Open-loop (free fall)")

        with st.expander("Physical parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                length = st.number_input("l [m]", value=1.0, min_value=0.0, key=f"{prefix}_length", help="Pole length from pivot to tip.")
                mass = st.number_input("m [kg]", value=0.2, min_value=0.0, key=f"{prefix}_mass", help="Pole (bob) mass.")
            with c2:
                cart_mass = st.number_input("M [kg]", value=0.5, min_value=0.0, key=f"{prefix}_cart_mass", help="Cart mass.")
                g = st.number_input("g [m/s²]", value=9.81, key=f"{prefix}_g", help="Gravitational acceleration.")
            with c3:
                # keep as an explicit control because it can be useful for numerical smoothing
                coulomb_k = st.number_input(
                    "Coulomb k",
                    value=1e3,
                    min_value=1.0,
                    format="%.3g",
                    key=f"{prefix}_coulomb_k",
                    help=(
                        "Smooth sign(v) ≈ tanh(k·v) used for Coulomb friction. "
                        "Higher k = sharper switch (more realistic, can be stiffer); "
                        "lower k = smoother (more stable)."
                    ),
                )

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
                    b_cart = st.number_input("b cart [N·m·s]", value=b_cart, min_value=0.0, key=f"{prefix}_b_cart", help="Viscous friction coefficient of the cart.")
                    coulomb_cart = st.number_input("Fc cart [N·m]", value=coulomb_cart, min_value=0.0, key=f"{prefix}_coulomb_cart", help="Coulomb (dry) friction force on the cart.")
                with f2:
                    b_pend = st.number_input("b pend [N·m·s]", value=b_pend, min_value=0.0, key=f"{prefix}_b_pend", help="Viscous friction coefficient at the pole pivot.")
                    coulomb_pend = st.number_input("Fc pend [N·m]", value=coulomb_pend, min_value=0.0, key=f"{prefix}_coulomb_pend", help="Coulomb (dry) friction torque at the pole pivot.")
        else:
            b_cart = coulomb_cart = b_pend = coulomb_pend = 0.0

        if mode == "driven":
            with st.expander("External drives", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    cart_drive_amp = st.number_input("Cart drive A", value=cart_drive_amp, min_value=0.0, key=f"{prefix}_cart_drive_amp", help="Amplitude of sinusoidal force applied to the cart [N].")
                    cart_drive_freq = st.number_input("Cart drive ω", value=cart_drive_freq, min_value=0.0, key=f"{prefix}_cart_drive_freq", help="Angular frequency of the cart drive force [rad/s].")
                    cart_drive_phase = st.number_input("Cart drive φ", value=cart_drive_phase, key=f"{prefix}_cart_drive_phase", help="Phase offset of the cart drive force [rad].")
                with d2:
                    pend_drive_amp = st.number_input("Pend drive A", value=pend_drive_amp, min_value=0.0, key=f"{prefix}_pend_drive_amp", help="Amplitude of sinusoidal torque applied to the pole [N·m].")
                    pend_drive_freq = st.number_input("Pend drive ω", value=pend_drive_freq, min_value=0.0, key=f"{prefix}_pend_drive_freq", help="Angular frequency of the pole drive torque [rad/s].")
                    pend_drive_phase = st.number_input("Pend drive φ", value=pend_drive_phase, key=f"{prefix}_pend_drive_phase", help="Phase offset of the pole drive torque [rad].")
        else:
            cart_drive_amp = cart_drive_freq = cart_drive_phase = 0.0
            pend_drive_amp = pend_drive_freq = pend_drive_phase = 0.0

        with st.expander("Initial state", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.number_input("x(0) [m]", value=0.0, format="%.6g", key=f"{prefix}_x0", help="Initial cart position.")
                xdot0 = st.number_input("ẋ(0) [m/s]", value=0.0, format="%.6g", key=f"{prefix}_xdot0", help="Initial cart velocity.")
            with c2:
                th0 = st.number_input("θ(0) [rad]", value=float(np.deg2rad(10.0)), format="%.6g", key=f"{prefix}_th0", help="Initial pole angle from upright (θ=0 is upright).")
                thdot0 = st.number_input("θ̇(0) [rad/s]", value=0.0, format="%.6g", key=f"{prefix}_thdot0", help="Initial pole angular velocity.")

        # controller settings
        lqr_settings: Dict[str, Any] = {}
        swing_settings: Dict[str, Any] = {}
        switch_settings: Dict[str, Any] = {}

        if ctrl_mode == "LQR stabilizer":
            lqr_settings = _lqr_settings_expander(prefix)

        elif ctrl_mode == "Swing-up only":
            swing_settings = _swingup_settings_expander(prefix)

        elif ctrl_mode == "Swing-up + LQR (simple)":
            swing_settings = _swingup_settings_expander(prefix)
            lqr_settings = _lqr_settings_expander(prefix)

            with st.expander("Switcher settings", expanded=False):
                s1, s2 = st.columns(2)
                with s1:
                    st.caption("Engage LQR when below:")
                    engage_angle_deg = st.number_input("Angle [°]", value=25.0, min_value=0.0, key=f"{prefix}_engage_angle_deg",
                                                       help="Switch to LQR when pole is within this angle of upright.")
                    engage_speed_rad_s = st.number_input("Pole speed [rad/s]", value=9.0, min_value=0.0, key=f"{prefix}_engage_speed_rad_s",
                                                          help="Switch to LQR only if angular speed is below this threshold.")
                    engage_cart_speed = st.number_input("Cart speed [m/s]", value=6.0, min_value=0.0, key=f"{prefix}_engage_cart_speed",
                                                         help="Switch to LQR only if cart speed is below this threshold.")
                with s2:
                    st.caption("Exit LQR when above:")
                    dropout_angle_deg = st.number_input("Angle [°] ", value=45.0, min_value=0.0, key=f"{prefix}_dropout_angle_deg",
                                                         help="Drop back to swing-up if pole exceeds this angle while in LQR mode.")
                    dropout_speed_rad_s = st.number_input("Pole speed [rad/s] ", value=30.0, min_value=0.0, key=f"{prefix}_dropout_speed_rad_s",
                                                           help="Drop back to swing-up if angular speed exceeds this while in LQR mode.")
                    dropout_cart_speed = st.number_input("Cart speed [m/s] ", value=10.0, min_value=0.0, key=f"{prefix}_dropout_cart_speed",
                                                          help="Drop back to swing-up if cart speed exceeds this while in LQR mode.")
                allow_dropout = st.checkbox("Allow LQR exit", value=True, key=f"{prefix}_allow_dropout",
                                            help="If checked, LQR gives up and returns to swing-up when the pole diverges too far.")
                b1, b2 = st.columns(2)
                with b1:
                    blend_time = st.number_input("Blend time [s]", value=0.12, min_value=0.0, key=f"{prefix}_blend_time",
                                                 help="Transition time when switching between controllers (if supported).")
                with b2:
                    du_max = st.number_input("Force rate [N/s]", value=800.0, min_value=0.0, key=f"{prefix}_du_max",
                                             help="Maximum rate of change of control force (if supported).")
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
