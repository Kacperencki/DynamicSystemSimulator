# apps/streamlit/inverted_runner.py

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models import get_model
from dss.wrappers.closed_lood_cart import CloseLoopCart
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
) -> Tuple[Dict, Dict]:
    """
    Open-loop inverted pendulum (no control).
    """
    system = get_model(
        "inverted_pendulum",
        mode=params["mode"],
        length=params["length"],
        mass=params["mass"],
        cart_mass=params["cart_mass"],
        gravity=params["g"],
        mass_model=params["mass_model"],
    )

    T_total = float(t1 - t0)

    # target based on dt
    if dt > 0:
        fps_target = 1.0 / dt
    else:
        fps_target = 100.0

    # hard cap on fps to avoid too many steps
    FPS_CAP = 200.0
    fps_eff = int(min(fps_target, FPS_CAP))
    if fps_eff < 1:
        fps_eff = 1

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]

    sol = Solver(system, initial_conditions=[x0, xdot0, th0, thdot0], T=T_total, fps=fps_eff).run()
    T = sol.t
    X = sol.y.T

    E_parts = np.array([system.energy_check(s) for s in X])
    KE, PE, E = E_parts[:, 0], E_parts[:, 1], E_parts[:, 2]

    cfg = dict(
        sys="ip_open",
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
) -> Tuple[Dict, Dict]:
    """
    Closed-loop inverted pendulum: plant + controller via CloseLoopCart.

    ctrl_mode:
        - "LQR stabilizer"
        - "Swing-up only"
        - "Swing-up + LQR (simple)"
    """
    # physical plant
    plant = get_model(
        "inverted_pendulum",
        mode=params["mode"],
        length=params["length"],
        mass=params["mass"],
        cart_mass=params["cart_mass"],
        gravity=params["g"],
        mass_model=params["mass_model"],
    )

    # ----- LQR parameter mapping (GUI → AutoLQR) -----------------
    # GUI gives: q_x, q_xdot, q_theta, q_thetad, r_u, u_max  :contentReference[oaicite:2]{index=2}
    # AutoLQR wants: x_max, xd_max, theta_max_deg, thetad_max, u_max :contentReference[oaicite:3]{index=3}

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
    # GUI gives: k_e, u_max  :contentReference[oaicite:4]{index=4}
    # AutoSwingUp wants: ke, kv, force_limit, ... :contentReference[oaicite:5]{index=5}

    def build_swing():
        k_e = swing_set.get("k_e", None)
        u_max = swing_set.get("u_max", None)
        return AutoSwingUp(
            plant,
            ke=k_e,
            force_limit=u_max,
        )

    # ----- Switcher mapping (GUI → SimpleSwitcher)  -------------- :contentReference[oaicite:6]{index=6}

    def build_switch(lqr, swing):
        return SimpleSwitcher(
            system=plant,
            lqr_controller=lqr,
            swingup_controller=swing,
            engage_angle_deg=float(switch_set.get("engage_angle_deg", 25.0)),
            engage_speed_rad_s=float(switch_set.get("engage_speed_rad_s", 9.0)),
            dropout_angle_deg=float(switch_set.get("dropout_angle_deg", 45.0)),
            allow_dropout=bool(switch_set.get("allow_dropout", True)),
            # keep the rest at good defaults; turn off spam:
            verbose=False,
        )

    # ----- Build controller by mode -------------------------------

    if ctrl_mode == "LQR stabilizer":
        lqr = build_lqr()
        controller = lqr

    elif ctrl_mode == "Swing-up only":
        swing = build_swing()
        controller = swing

    elif ctrl_mode == "Swing-up + LQR (simple)":
        lqr = build_lqr()
        swing = build_swing()
        controller = build_switch(lqr, swing)
    else:
        raise ValueError(f"Unsupported ctrl_mode: {ctrl_mode!r}")

    closed = CloseLoopCart(system=plant, controller=controller)

    T_total = float(t1 - t0)

    # target based on dt
    if dt > 0:
        fps_target = 1.0 / dt
    else:
        fps_target = 100.0

    # hard cap on fps to avoid too many steps
    FPS_CAP = 200.0
    fps_eff = int(min(fps_target, FPS_CAP))
    if fps_eff < 1:
        fps_eff = 1

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]
    sol = Solver(closed, initial_conditions=[x0, xdot0, th0, thdot0], T=T_total, fps=fps_eff).run()

    T = sol.t
    X = sol.y.T

    E_parts = np.array([closed.energy_check(s) for s in X])
    KE, PE, E = E_parts[:, 0], E_parts[:, 1], E_parts[:, 2]

    cfg = dict(
        sys="ip_closed",
        ctrl_mode=ctrl_mode,
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
