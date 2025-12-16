# apps/streamlit/runners/lorenz_runner.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models.lorenz import Lorenz
from dss.core.solver import Solver


def run_lorenz(params: Dict, ic: Dict, t0: float, t1: float, dt: float) -> Tuple[Dict, Dict]:
    sigma = float(params["sigma"])
    rho = float(params["rho"])
    beta = float(params["beta"])

    x0 = float(ic["x0"])
    y0 = float(ic["y0"])
    z0 = float(ic["z0"])

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 200

    system = Lorenz(sigma=sigma, rho=rho, beta=beta)
    sol = Solver(system, initial_conditions=[x0, y0, z0], T=T_total, fps=fps_eff).run()

    T = np.asarray(sol.t, dtype=float)
    X = np.asarray(sol.y.T, dtype=float)

    cfg = dict(
        sys="lorenz",
        sigma=sigma,
        rho=rho,
        beta=beta,
        x0=x0,
        y0=y0,
        z0=z0,
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
    )
    out = dict(T=T, X=X)
    return cfg, out
