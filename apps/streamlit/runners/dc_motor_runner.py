# apps/streamlit/runners/dc_motor_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from apps.streamlit.runners._common import run_from_cfg


def _voltage_profile(t: np.ndarray, mode: str, p: Dict) -> np.ndarray:
    V0 = float(p.get("V0", 0.0))
    off = float(p.get("v_offset", 0.0))
    t_step = float(p.get("t_step", 0.0))
    freq = float(p.get("v_freq", 1.0))
    duty = float(p.get("v_duty", 0.5))

    if mode == "step":
        return off + (t >= t_step).astype(float) * V0
    if mode == "sine":
        return off + V0 * np.sin(2.0 * np.pi * freq * t)
    if mode == "square":
        phase = (freq * t) % 1.0
        return off + V0 * (phase < duty).astype(float)
    return off + 0.0 * t


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
    save_run: bool = False,
    log_dir: str = "logs",
    run_name: str = "",
) -> Tuple[Dict, Dict]:
    i0 = float(ic["i0"])
    omega0 = float(ic["omega0"])

    # Pass all motor params through; the model supports both runner-style fields and legacy hooks.
    cfg = {
        "model": {"name": "dc_motor", "mode": "default", "params": dict(params)},
        "initial_state": [i0, omega0],
        "solver": {"t0": float(t0), "t1": float(t1), "dt": float(dt), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    }

    cfg2, out = run_from_cfg(cfg, save_run=save_run, log_dir=log_dir, run_name=run_name)

    T = np.asarray(out["T"])
    X = np.asarray(out["X"])

    # Derived signals for dashboard: applied voltage and angle theta(t)=∫omega dt
    v_mode = str(params.get("v_mode", "step"))
    V = _voltage_profile(T, v_mode, params)

    omega = X[:, 1]
    theta = np.zeros_like(omega)
    if len(T) >= 2:
        dt_local = np.diff(T)
        theta[1:] = np.cumsum(0.5 * (omega[1:] + omega[:-1]) * dt_local)

    out["V"] = V
    out["theta"] = theta
    return cfg2, out
