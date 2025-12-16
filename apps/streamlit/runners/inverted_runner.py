# apps/streamlit/runners/inverted_runner.py

from __future__ import annotations

from typing import Dict, Tuple
import inspect

import numpy as np

from dss.models import get_model
from dss.wrappers.closed_loop_cart import ClosedLoopCart
from dss.controllers.lqr_controller import AutoLQR
from dss.controllers.swingup import AutoSwingUp
from dss.controllers.simple_switcher import SimpleSwitcher
from dss.core.solver import Solver


def run_ip_open(
    params: Dict,
    ic: Dict,
    t0: float,
    t1: float,
    dt: float,
    *,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Tuple[Dict, Dict]:
    """Open-loop inverted pendulum (no control)."""
    system = get_model(
        "inverted_pendulum",
        mode=params["mode"],
        length=params["length"],
        mass=params["mass"],
        cart_mass=params["cart_mass"],
        gravity=params["g"],
        mass_model=params.get("mass_model", "point"),
        # friction
        b_cart=params.get("b_cart", 0.0),
        coulomb_cart=params.get("coulomb_cart", 0.0),
        b_pend=params.get("b_pend", 0.0),
        coulomb_pend=params.get("coulomb_pend", 0.0),
        coulomb_k=params.get("coulomb_k", 1e3),
        # harmonic drives
        cart_drive_amp=params.get("cart_drive_amp", 0.0),
        cart_drive_freq=params.get("cart_drive_freq", 0.0),
        cart_drive_phase=params.get("cart_drive_phase", 0.0),
        pend_drive_amp=params.get("pend_drive_amp", 0.0),
        pend_drive_freq=params.get("pend_drive_freq", 0.0),
        pend_drive_phase=params.get("pend_drive_phase", 0.0),
    )

    T_total = float(t1 - t0)

    # target based on dt
    fps_target = 1.0 / dt if dt > 0 else 100.0

    # hard cap on fps to avoid too many steps
    FPS_CAP = 200.0
    fps_eff = int(min(fps_target, FPS_CAP))
    if fps_eff < 1:
        fps_eff = 1

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]

    sol = Solver(
        system,
        initial_conditions=[x0, xdot0, th0, thdot0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T

    E_parts = np.array([system.energy_check(s) for s in X])
    KE, PE, E = E_parts[:, 0], E_parts[:, 1], E_parts[:, 2]

    cfg = dict(
        sys="ip_open",
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        **params,
        x0=x0,
        xdot0=xdot0,
        th0=th0,
        thdot0=thdot0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X, E_parts=(KE, PE, E))
    return cfg, out


def run_ip_closed(
    ctrl_mode: str,
    params: Dict,
    lqr_set: Dict,
    swing_set: Dict,
    switch_set: Dict,
    ic: Dict,
    t0: float,
    t1: float,
    dt: float,
    *,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Tuple[Dict, Dict]:
    """Closed-loop inverted pendulum: plant + controller via ClosedLoopCart."""
    # physical plant
    plant = get_model(
        "inverted_pendulum",
        mode=params["mode"],
        length=params["length"],
        mass=params["mass"],
        cart_mass=params["cart_mass"],
        gravity=params["g"],
        mass_model=params.get("mass_model", "point"),
        # friction
        b_cart=params.get("b_cart", 0.0),
        coulomb_cart=params.get("coulomb_cart", 0.0),
        b_pend=params.get("b_pend", 0.0),
        coulomb_pend=params.get("coulomb_pend", 0.0),
        coulomb_k=params.get("coulomb_k", 1e3),
        # harmonic drives
        cart_drive_amp=params.get("cart_drive_amp", 0.0),
        cart_drive_freq=params.get("cart_drive_freq", 0.0),
        cart_drive_phase=params.get("cart_drive_phase", 0.0),
        pend_drive_amp=params.get("pend_drive_amp", 0.0),
        pend_drive_freq=params.get("pend_drive_freq", 0.0),
        pend_drive_phase=params.get("pend_drive_phase", 0.0),
    )

    # ----- LQR parameter mapping (GUI → AutoLQR) -----------------
    def build_lqr():
        q_x = float(lqr_set.get("q_x", 1.0))
        q_xdot = float(lqr_set.get("q_xdot", 1.0))
        q_theta = float(lqr_set.get("q_theta", 50.0))
        q_thetad = float(lqr_set.get("q_thetad", 1.0))
        u_max = float(lqr_set.get("u_max", 20.0))

        # Base "allowed magnitudes"
        base_x_max = 0.25
        base_xd_max = 2.0
        base_theta_max_deg = 8.0
        base_thetad_max = 4.0

        # Simple monotonic mapping: higher q -> smaller allowed magnitude
        x_scale = np.sqrt(max(q_x, 1e-3) / 1.0)
        xd_scale = np.sqrt(max(q_xdot, 1e-3) / 1.0)
        th_scale = np.sqrt(max(q_theta, 1e-3) / 50.0)
        thd_scale = np.sqrt(max(q_thetad, 1e-3) / 1.0)

        x_max = float(np.clip(base_x_max / x_scale, 0.05, 1.0))
        xd_max = float(np.clip(base_xd_max / xd_scale, 0.2, 5.0))
        theta_max_deg = float(np.clip(base_theta_max_deg / th_scale, 2.0, 30.0))
        thetad_max = float(np.clip(base_thetad_max / thd_scale, 0.5, 10.0))

        return AutoLQR(
            plant,
            x_max=x_max,
            xd_max=xd_max,
            theta_max_deg=theta_max_deg,
            thetad_max=thetad_max,
            u_max=u_max,
        )

    # ----- Swing-up parameter mapping (GUI → AutoSwingUp) --------
    def build_swing():
        k_e = swing_set.get("k_e", None)
        u_max = swing_set.get("u_max", None)
        return AutoSwingUp(
            plant,
            ke=k_e,
            force_limit=u_max,
        )

    # ----- Switcher mapping (GUI → SimpleSwitcher)  --------------
    def build_switch(lqr, swing):
        raw_kwargs = dict(
            system=plant,
            lqr_controller=lqr,
            swingup_controller=swing,
            engage_angle_deg=float(switch_set.get("engage_angle_deg", 25.0)),
            engage_speed_rad_s=float(switch_set.get("engage_speed_rad_s", 9.0)),
            engage_cart_speed=float(switch_set.get("engage_cart_speed", 6.0)),
            dropout_angle_deg=float(switch_set.get("dropout_angle_deg", 45.0)),
            dropout_speed_rad_s=float(switch_set.get("dropout_speed_rad_s", 30.0)),
            dropout_cart_speed=float(switch_set.get("dropout_cart_speed", 10.0)),
            allow_dropout=bool(switch_set.get("allow_dropout", True)),
            # smooth transitions (optional; kept if supported by the installed class)
            blend_time=float(switch_set.get("blend_time", 0.12)),
            du_max=float(switch_set.get("du_max", 800.0)),
            verbose=False,
        )
        sig = inspect.signature(SimpleSwitcher.__init__)
        kwargs = {k: v for k, v in raw_kwargs.items() if k in sig.parameters}
        return SimpleSwitcher(**kwargs)

    # ----- Build controller by mode -------------------------------
    if ctrl_mode == "LQR stabilizer":
        lqr = build_lqr()
        # Stabilize around the current cart position by default (prevents snapping to x=0)
        if hasattr(lqr, "x_ref"):
            lqr.x_ref = float(ic.get("x0", 0.0))
        controller = lqr

    elif ctrl_mode == "Swing-up only":
        swing = build_swing()
        controller = swing

    elif ctrl_mode == "Swing-up + LQR (simple)":
        lqr = build_lqr()
        # Same idea: when LQR engages, it should hold the current x as reference
        # (the switcher also sets x_ref on ENTER, but this keeps things consistent)
        if hasattr(lqr, "x_ref"):
            lqr.x_ref = float(ic.get("x0", 0.0))
        swing = build_swing()
        controller = build_switch(lqr, swing)

    else:
        raise ValueError(f"Unsupported ctrl_mode: {ctrl_mode!r}")

    closed = ClosedLoopCart(system=plant, controller=controller)

    T_total = float(t1 - t0)

    fps_target = 1.0 / dt if dt > 0 else 100.0
    FPS_CAP = 200.0
    fps_eff = int(min(fps_target, FPS_CAP))
    if fps_eff < 1:
        fps_eff = 1

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]
    sol = Solver(
        closed,
        initial_conditions=[x0, xdot0, th0, thdot0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()

    T = sol.t
    X = sol.y.T

    E_parts = np.array([closed.energy_check(s) for s in X])
    KE, PE, E = E_parts[:, 0], E_parts[:, 1], E_parts[:, 2]

    cfg = dict(
        sys="ip_closed",
        ctrl_mode=ctrl_mode,
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        **params,
        x0=x0,
        xdot0=xdot0,
        th0=th0,
        thdot0=thdot0,
        t0=t0,
        t1=t1,
        dt=dt,
        lqr_settings=lqr_set,
        swing_settings=swing_set,
        switch_settings=switch_set,
    )
    out = dict(T=T, X=X, E_parts=(KE, PE, E))
    return cfg, out
