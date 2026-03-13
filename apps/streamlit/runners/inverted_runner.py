# apps/streamlit/runners/inverted_runner.py
"""
Simulation runners for the inverted pendulum / cart-pole system.

Exported functions
------------------
run_ip_open(params, ic, t0, t1, dt, ...)
    Open-loop: plant dynamics only, no control force.

run_ip_closed(ctrl_mode, params, lqr_set, swing_set, switch_set, ic, t0, t1, dt, ...)
    Closed-loop: plant wrapped with controller.
    ctrl_mode selects the controller type:
      "LQR stabilizer"       → AutoLQR only
      "Swing-up only"        → AutoSwingUp only
      "Swing-up + LQR ..."   → SimpleSwitcher (swing-up → LQR handoff)

LQR weight mapping (UI → Bryson magnitudes)
-------------------------------------------
The UI exposes intuitive "weight" sliders (q_x, q_theta, …) rather than the
raw Bryson allowed-amplitude values.  The mapping is:
    allowed_amplitude = base_amplitude / sqrt(weight / base_weight)

So:  weight ↑  →  allowed amplitude ↓  →  stronger correction of that state.
Defaults (base weights): q_x=1, q_xdot=1, q_theta=60, q_thetad=1.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from dss.models import get_model
from dss.wrappers.closed_loop_cart import ClosedLoopCart
from dss.controllers.lqr_controller import AutoLQR
from dss.controllers.swingup import AutoSwingUp
from dss.controllers.simple_switcher import SimpleSwitcher

from apps.streamlit.runners._common import run_from_system


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_mode(mode: str) -> str:
    """Map legacy UI mode name to the model's internal mode string."""
    # Older UI offered a generic "damped" option; model expects "damped_both"
    if mode == "damped":
        return "damped_both"
    return mode


def _plant_from_params(params: Dict[str, Any]):
    """Build the InvertedPendulum plant from a Streamlit controls dict.

    Accepts both new names (length/mass/cart_mass) and legacy names (l/m/M)
    so old presets / saved configs continue to work.
    """
    mode = _coerce_mode(str(params.get("mode", "damped_both")))

    # Parameter resolution order: new name → legacy short → fallback default
    length = float(params.get("length", params.get("l", params.get("L", 0.3))))
    mass = float(params.get("mass", params.get("m", 0.2)))
    cart_mass = float(params.get("cart_mass", params.get("M", 0.5)))
    g = float(params.get("g", params.get("gravity", 9.81)))

    plant = get_model(
        "inverted_pendulum",
        mode=mode,
        length=length,
        mass=mass,
        cart_mass=cart_mass,
        g=g,
        mass_model=str(params.get("mass_model", "point")),
        # Viscous and Coulomb friction for cart and pole
        b_cart=float(params.get("b_cart", 0.0)),
        coulomb_cart=float(params.get("coulomb_cart", 0.0)),
        b_pend=float(params.get("b_pend", 0.0)),
        coulomb_pend=float(params.get("coulomb_pend", 0.0)),
        coulomb_k=float(params.get("coulomb_k", 1e3)),   # tanh smoothing gain for Coulomb sign
        # Optional harmonic external drives (cart force and pole torque)
        cart_drive_amp=float(params.get("cart_drive_amp", 0.0)),
        cart_drive_freq=float(params.get("cart_drive_freq", 0.0)),
        cart_drive_phase=float(params.get("cart_drive_phase", 0.0)),
        pend_drive_amp=float(params.get("pend_drive_amp", 0.0)),
        pend_drive_freq=float(params.get("pend_drive_freq", 0.0)),
        pend_drive_phase=float(params.get("pend_drive_phase", 0.0)),
    )
    return plant, mode


# ---------------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------------

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
    save_run: bool = False,
    log_dir: str = "logs",
    run_name: str = "",
) -> Tuple[Dict, Dict]:
    """Open-loop simulation: no control force applied to the cart."""
    plant, mode = _plant_from_params(params)

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]

    # Cap fps to 200 Hz to avoid unnecessarily large output arrays
    FPS_CAP = 200.0
    dt_eff = max(float(dt), 1.0 / FPS_CAP)

    # Config dict stored alongside simulation output for reproducibility / logging
    cfg = dict(
        sys="ip_open",
        model={"name": "inverted_pendulum", "mode": mode, "params": dict(params)},
        initial_state=[float(x0), float(xdot0), float(th0), float(thdot0)],
        solver={"t0": float(t0), "t1": float(t1), "dt": float(dt_eff), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    )

    cfg2, out = run_from_system(
        plant,
        np.array([x0, xdot0, th0, thdot0], dtype=float),
        cfg,
        save_run=save_run,
        log_dir=log_dir,
        run_name=run_name,
    )

    # Compute energy components [T, V, E] at each time step for the dashboard
    X = np.asarray(out["X"])
    E_parts = np.array([plant.energy_check(s) for s in X])
    out["E_parts"] = E_parts
    return cfg2, out


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
    save_run: bool = False,
    log_dir: str = "logs",
    run_name: str = "",
) -> Tuple[Dict, Dict]:
    """Closed-loop simulation: plant + controller (LQR, swing-up, or switcher)."""

    plant, mode = _plant_from_params(params)

    # ------------------------------------------------------------------
    # Controller factory functions (defined here to close over `plant`)
    # ------------------------------------------------------------------

    def build_lqr() -> AutoLQR:
        """Convert UI weight sliders to Bryson allowed-amplitude values for AutoLQR."""
        q_x = float(lqr_set.get("q_x", 1.0))
        q_xdot = float(lqr_set.get("q_xdot", 1.0))
        q_theta = float(lqr_set.get("q_theta", 60.0))
        q_thetad = float(lqr_set.get("q_thetad", 1.0))
        u_max = float(lqr_set.get("u_max", 20.0))

        # Reference (base) allowed amplitudes at default weights
        base_x_max = 0.25          # [m]     cart position
        base_xd_max = 2.0          # [m/s]   cart velocity
        base_theta_max_deg = 8.0   # [deg]   pole angle
        base_thetad_max = 4.0      # [rad/s] pole angular velocity

        # Monotonic mapping: weight ↑  →  scale ↑  →  allowed amplitude ↓  →  stronger gain
        x_scale = np.sqrt(max(q_x, 1e-6) / 1.0)
        xd_scale = np.sqrt(max(q_xdot, 1e-6) / 1.0)
        th_scale = np.sqrt(max(q_theta, 1e-6) / 60.0)   # normalised to default q_theta=60
        thd_scale = np.sqrt(max(q_thetad, 1e-6) / 1.0)

        x_max = base_x_max / x_scale
        xd_max = base_xd_max / xd_scale
        theta_max_deg = base_theta_max_deg / th_scale
        thetad_max = base_thetad_max / thd_scale

        lqr = AutoLQR(
            plant,
            x_max=float(lqr_set.get("x_max", x_max)),
            xd_max=float(lqr_set.get("xd_max", xd_max)),
            theta_max_deg=float(lqr_set.get("theta_max_deg", theta_max_deg)),
            thetad_max=float(lqr_set.get("thetad_max", thetad_max)),
            u_max=u_max,
        )
        # Allow optional explicit references (used when chaining with switcher)
        if "x_ref" in lqr_set:
            lqr.x_ref = float(lqr_set["x_ref"])
        if "theta_ref" in lqr_set:
            lqr.theta_ref = float(lqr_set["theta_ref"])
        return lqr

    def build_swing() -> AutoSwingUp:
        """Build energy-based swing-up controller.

        ke=0 in the UI means "use physics-based default" (5 / (m·lc)).
        """
        ke_raw = swing_set.get("k_e", swing_set.get("ke", None))
        ke_float = float(ke_raw) if ke_raw is not None else 0.0
        ke = None if ke_float == 0.0 else ke_float   # None → AutoSwingUp computes default
        u_raw = swing_set.get("u_max", swing_set.get("force_limit", None))
        u_max = float(u_raw) if u_raw is not None else 25.0
        return AutoSwingUp(plant, ke=ke, force_limit=u_max)

    def build_switch(lqr: AutoLQR, swing: AutoSwingUp) -> SimpleSwitcher:
        """Build hard swing-up ↔ LQR switcher from UI settings."""
        return SimpleSwitcher(
            plant,
            lqr_controller=lqr,
            swingup_controller=swing,
            engage_angle_deg=float(switch_set.get("engage_angle_deg", 25.0)),
            engage_speed_rad_s=float(switch_set.get("engage_speed_rad_s", 9.0)),
            engage_cart_speed=float(switch_set.get("engage_cart_speed", 1.2)),
            dropout_angle_deg=float(switch_set.get("dropout_angle_deg", 110.0)),
            dropout_speed_rad_s=float(switch_set.get("dropout_speed_rad_s", 30.0)),
            dropout_cart_speed=float(switch_set.get("dropout_cart_speed", 10.0)),
            hold_time=float(switch_set.get("hold_time", 0.05)),
        )

    # ------------------------------------------------------------------
    # Select and build controller based on UI mode string
    # ------------------------------------------------------------------
    if ctrl_mode == "LQR stabilizer":
        controller = build_lqr()
        ctrl_name = "ip_lqr"
        ctrl_params = dict(lqr_set)
    elif ctrl_mode == "Swing-up only":
        controller = build_swing()
        ctrl_name = "ip_swingup"
        ctrl_params = dict(swing_set)
    elif ctrl_mode == "Swing-up + LQR (simple)":
        lqr = build_lqr()
        swing = build_swing()
        controller = build_switch(lqr, swing)
        ctrl_name = "ip_switch_simple"
        ctrl_params = {"lqr": dict(lqr_set), "swing": dict(swing_set), "switch": dict(switch_set)}
    else:
        raise ValueError(f"Unsupported ctrl_mode: {ctrl_mode!r}")

    # Wrap plant + controller: ClosedLoopCart.dynamics() calls controller internally
    closed = ClosedLoopCart(system=plant, controller=controller)

    x0, xdot0, th0, thdot0 = ic["x0"], ic["xdot0"], ic["th0"], ic["thdot0"]

    FPS_CAP = 200.0
    dt_eff = max(float(dt), 1.0 / FPS_CAP)

    cfg = dict(
        sys="ip_closed",
        model={"name": "inverted_pendulum", "mode": mode, "params": dict(params)},
        controller={"name": ctrl_name, "params": ctrl_params},
        wrapper={"name": "closed_loop_cart"},
        initial_state=[float(x0), float(xdot0), float(th0), float(thdot0)],
        solver={"t0": float(t0), "t1": float(t1), "dt": float(dt_eff), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    )

    cfg2, out = run_from_system(
        closed,
        np.array([x0, xdot0, th0, thdot0], dtype=float),
        cfg,
        save_run=save_run,
        log_dir=log_dir,
        run_name=run_name,
    )

    # Evaluate plant energy (not closed-loop wrapper) at each stored time step
    X = np.asarray(out["X"])
    E_parts = np.array([plant.energy_check(s) for s in X])
    out["E_parts"] = E_parts
    return cfg2, out
