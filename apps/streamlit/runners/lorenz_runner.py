# apps/streamlit/runners/lorenz_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from apps.streamlit.runners._common import run_from_cfg


def run_lorenz(
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
    sigma = float(params["sigma"])
    rho = float(params["rho"])
    beta = float(params["beta"])

    x0 = float(ic["x0"])
    y0 = float(ic["y0"])
    z0 = float(ic["z0"])

    cfg = {
        "model": {"name": "lorenz", "mode": "default", "params": {"sigma": sigma, "rho": rho, "beta": beta}},
        "initial_state": [x0, y0, z0],
        "solver": {"t0": float(t0), "t1": float(t1), "dt": float(dt), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    }

    cfg2, out = run_from_cfg(cfg, save_run=save_run, log_dir=log_dir, run_name=run_name)
    return cfg2, out

