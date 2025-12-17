# apps/streamlit/runners/pendulum_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from apps.streamlit.runners._common import run_from_cfg


def run_single_pendulum(
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
    mode = str(params.get("mode", "damped"))
    cfg = {
        "model": {"name": "pendulum", "mode": mode, "params": dict(params)},
        "initial_state": [float(ic["theta0"]), float(ic["omega0"])],
        "solver": {"t0": float(t0), "t1": float(t1), "dt": float(dt), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    }

    cfg2, out = run_from_cfg(cfg, save_run=save_run, log_dir=log_dir, run_name=run_name)

    # Energy parts for dashboards / experiments
    X = np.asarray(out["X"])
    # model instance is not returned here; compute via formula when possible? keep runner-style loop using get_model
    # For now we keep this optional; dashboards do not require it.
    return cfg2, out


def run_double_pendulum(
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
    mode = str(params.get("mode", "damped"))
    cfg = {
        "model": {"name": "double_pendulum", "mode": mode, "params": dict(params)},
        "initial_state": [
            float(ic.get("th1_0", 0.0)),
            float(ic.get("th1d_0", ic.get("w1_0", 0.0))),
            float(ic.get("th2_0", 0.0)),
            float(ic.get("th2d_0", ic.get("w2_0", 0.0))),
        ],
        "solver": {"t0": float(t0), "t1": float(t1), "dt": float(dt), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    }
    cfg2, out = run_from_cfg(cfg, save_run=save_run, log_dir=log_dir, run_name=run_name)
    return cfg2, out
