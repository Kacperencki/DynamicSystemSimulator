# apps/streamlit/runners/vanderpol_runner.py

from __future__ import annotations

from typing import Dict, Tuple

from apps.streamlit.runners._common import run_from_cfg


def run_vanderpol(
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
    mu = float(params.get("mu", 1.0))
    v0 = float(ic["v0"])
    iL0 = float(ic["iL0"])

    cfg = {
        "model": {"name": "vanderpol", "mode": "default", "params": {"mu": mu}},
        "initial_state": [v0, iL0],
        "solver": {"t0": float(t0), "t1": float(t1), "dt": float(dt), "method": str(method), "rtol": float(rtol), "atol": float(atol)},
    }

    cfg2, out = run_from_cfg(cfg, save_run=save_run, log_dir=log_dir, run_name=run_name)
    return cfg2, out
