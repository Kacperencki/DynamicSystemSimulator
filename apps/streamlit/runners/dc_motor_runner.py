# apps/streamlit/runners/dc_motor_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dss.models.dc_motor import DCMotor
from dss.core.solver import Solver


def _voltage_profile(t: np.ndarray, mode: str, p: Dict) -> np.ndarray:
    V0 = float(p.get("V0", 0.0))
    off = float(p.get("v_offset", 0.0))
    t_step = float(p.get("t_step", 0.0))
    f = float(p.get("v_freq", 1.0))
    duty = float(p.get("v_duty", 0.5))

    if mode == "constant":
        return off + V0 * np.ones_like(t)
    if mode == "step":
        return off + V0 * (t >= t_step).astype(float)
    if mode == "ramp":
        return off + V0 * np.clip((t - t_step), 0.0, None)
    if mode == "sine":
        return off + V0 * np.sin(2.0 * np.pi * f * t)
    if mode == "square":
        phase = (f * t) % 1.0
        return off + V0 * (phase < duty).astype(float)
    return off + V0 * np.ones_like(t)


def run_dc_motor(
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
    R = float(params["R"])
    L = float(params["L"])
    Ke = float(params["Ke"])
    Kt = float(params["Kt"])
    J = float(params["J"])
    bm = float(params["bm"])

    v_mode = str(params["v_mode"])
    V0 = float(params.get("V0", 0.0))
    v_offset = float(params.get("v_offset", 0.0))
    t_step = float(params.get("t_step", 0.05))
    v_freq = float(params.get("v_freq", 1.0))
    v_duty = float(params.get("v_duty", 0.5))

    load_mode = str(params.get("load_mode", "none"))
    tau_load = float(params.get("tau_load", 0.0))
    b_load = float(params.get("b_load", 0.0))
    tau_c = float(params.get("tau_c", 0.0))
    omega_eps = float(params.get("omega_eps", 0.5))

    i0 = float(ic["i0"])
    omega0 = float(ic["omega0"])

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    motor = DCMotor(
        R=R,
        L=L,
        Ke=Ke,
        Kt=Kt,
        J=J,
        bm=bm,
        v_mode=v_mode,
        V0=V0,
        v_offset=v_offset,
        t_step=t_step,
        v_freq=v_freq,
        v_duty=v_duty,
        load_mode=load_mode,
        tau_load=tau_load,
        b_load=b_load,
        tau_c=tau_c,
        omega_eps=omega_eps,
    )

    sol = Solver(
        motor,
        initial_conditions=[i0, omega0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T

    # Derived signals
    V = _voltage_profile(T, v_mode, dict(V0=V0, v_offset=v_offset, t_step=t_step, v_freq=v_freq, v_duty=v_duty))
    theta = np.cumsum(X[:, 1]) * (float(np.mean(np.diff(T))) if len(T) > 1 else 0.0)

    cfg = dict(
        sys="dcm",
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        R=R,
        L=L,
        Ke=Ke,
        Kt=Kt,
        J=J,
        bm=bm,
        v_mode=v_mode,
        V0=V0,
        v_offset=v_offset,
        t_step=t_step,
        v_freq=v_freq,
        v_duty=v_duty,
        load_mode=load_mode,
        tau_load=tau_load,
        b_load=b_load,
        tau_c=tau_c,
        omega_eps=omega_eps,
        i0=i0,
        omega0=omega0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X, V=V, theta=theta)
    return cfg, out
