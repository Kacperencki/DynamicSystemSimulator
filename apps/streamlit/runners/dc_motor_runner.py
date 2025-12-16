# apps/streamlit/runners/dc_motor_runner.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models.dc_motor import DCMotor
from dss.core.solver import Solver


def _make_voltage_fn(mode: str, V0: float, t_step: float, freq: float, duty: float, offset: float) -> Callable[[float], float]:
    mode = str(mode)
    V0 = float(V0)
    t_step = float(t_step)
    freq = float(freq)
    duty = float(duty)
    offset = float(offset)

    if mode == "constant":
        return lambda t: V0 + offset

    if mode == "step":
        return lambda t: (0.0 + offset) if t < t_step else (V0 + offset)

    if mode == "sine":
        w = 2.0 * np.pi * freq
        return lambda t: offset + V0 * np.sin(w * t)

    if mode == "square":
        # simple duty-based square wave around offset
        T = 1.0 / freq if freq > 0 else 1.0
        duty = min(max(duty, 0.0), 1.0)
        return lambda t: offset + (V0 if ( (t % T) < (duty * T) ) else -V0)

    if mode == "ramp":
        # ramp from 0 to V0 over t_step
        t_r = max(t_step, 1e-9)
        return lambda t: offset + (V0 * min(max(t / t_r, 0.0), 1.0))

    # fallback
    return lambda t: V0 + offset


def _make_load_fn(mode: str, tau_const: float, b_load: float, tau_c: float, omega_eps: float) -> Callable[[float, float], float]:
    mode = str(mode)
    tau_const = float(tau_const)
    b_load = float(b_load)
    tau_c = float(tau_c)
    omega_eps = float(omega_eps)

    if mode == "none":
        return lambda t, omega: 0.0

    if mode == "constant":
        return lambda t, omega: tau_const

    if mode == "viscous":
        return lambda t, omega: b_load * float(omega)

    if mode == "coulomb":
        # smooth sign to avoid hard switching at 0
        eps = max(omega_eps, 1e-9)
        return lambda t, omega: tau_c * float(np.tanh(float(omega) / eps))

    return lambda t, omega: 0.0


def run_dc_motor(params: Dict, ic: Dict, t0: float, t1: float, dt: float) -> Tuple[Dict, Dict]:
    R = float(params["R"])
    L = float(params["L"])
    Ke = float(params["Ke"])
    Kt = float(params["Kt"])
    J = float(params["J"])
    bm = float(params["bm"])

    v_mode = str(params.get("v_mode", "step"))
    V0 = float(params.get("V0", 6.0))
    v_offset = float(params.get("v_offset", 0.0))
    t_step = float(params.get("t_step", 0.05))
    v_freq = float(params.get("v_freq", 1.0))
    v_duty = float(params.get("v_duty", 0.5))

    load_mode = str(params.get("load_mode", "none"))
    tau_load = float(params.get("tau_load", 0.0))
    b_load = float(params.get("b_load", 0.0))
    tau_c = float(params.get("tau_c", 0.0))
    omega_eps = float(params.get("omega_eps", 0.5))

    i0 = float(ic.get("i0", 0.0))
    omega0 = float(ic.get("omega0", 0.0))

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 300

    v_fn = _make_voltage_fn(v_mode, V0, t_step, v_freq, v_duty, v_offset)
    load_fn = _make_load_fn(load_mode, tau_load, b_load, tau_c, omega_eps)

    motor = DCMotor(
        R=R,
        L=L,
        Ke=Ke,
        Kt=Kt,
        Im=J,
        bm=bm,
        voltage_func=v_fn,
        load_func=load_fn,
    )

    sol = Solver(motor, initial_conditions=[i0, omega0], T=T_total, fps=fps_eff).run()

    T = np.asarray(sol.t, dtype=float)
    X = np.asarray(sol.y.T, dtype=float)

    i = X[:, 0]
    omega = X[:, 1]

    V = np.asarray([float(v_fn(float(t))) for t in T], dtype=float)
    emf = Ke * omega
    tau = Kt * i

    # derived angle for animation
    if len(T) > 1:
        dT = np.diff(T)
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (omega[:-1] + omega[1:]) * dT)])
    else:
        theta = np.zeros_like(T)

    cfg = dict(
        sys="dc_motor",
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
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
    )
    out = dict(T=T, X=X, V=V, emf=emf, tau=tau, theta=theta)
    return cfg, out
