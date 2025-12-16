# apps/streamlit/runners/lorenz_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dss.models.lorenz import Lorenz
from dss.core.solver import Solver


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
) -> Tuple[Dict, Dict]:
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    x0, y0, z0 = ic["x0"], ic["y0"], ic["z0"]

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    sys = Lorenz(sigma=sigma, rho=rho, beta=beta)

    sol = Solver(
        sys,
        initial_conditions=[x0, y0, z0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T

    cfg = dict(
        sys="lorenz",
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        sigma=sigma,
        rho=rho,
        beta=beta,
        x0=x0,
        y0=y0,
        z0=z0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X)
    return cfg, out
