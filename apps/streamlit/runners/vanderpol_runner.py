# apps/streamlit/runners/vanderpol_runner.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss.models import get_model
from dss.core.solver import Solver


def run_vanderpol(params: Dict, ic: Dict, t0: float, t1: float, dt: float) -> Tuple[Dict, Dict]:
    """Run Van der Pol (nonlinear LC) simulation.

    Model states: v [V], iL [A]
    """
    L = float(params["L"])
    C = float(params["C"])
    mu = float(params["mu"])

    v0 = float(ic["v0"])
    iL0 = float(ic["iL0"])

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 100

    system = get_model("vanderpol", mode="default", L=L, C=C, mu=mu)
    sol = Solver(system, initial_conditions=[v0, iL0], T=T_total, fps=fps_eff).run()

    T = np.asarray(sol.t, dtype=float)
    X = np.asarray(sol.y.T, dtype=float)

    # Useful derived signal: dv/dt (exact from model equation)
    v = X[:, 0]
    iL = X[:, 1]
    i_nl = mu * ((v ** 3) / 3.0 - v)
    dv_dt = (-iL - i_nl) / C

    cfg = dict(
        sys="vanderpol",
        L=L,
        C=C,
        mu=mu,
        v0=v0,
        iL0=iL0,
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
    )
    out = dict(T=T, X=X, dv_dt=np.asarray(dv_dt, dtype=float))
    return cfg, out
