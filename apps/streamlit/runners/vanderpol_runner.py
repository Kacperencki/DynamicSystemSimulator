# apps/streamlit/runners/vanderpol_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dss.models.vanderpol_circuit import VanDerPol
from dss.core.solver import Solver


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
) -> Tuple[Dict, Dict]:
    L, C, mu = params["L"], params["C"], params["mu"]
    v0, iL0 = ic["v0"], ic["iL0"]

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    sys = VanDerPol(L=L, C=C, mu=mu)

    sol = Solver(
        sys,
        initial_conditions=[v0, iL0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T

    if len(T) > 1:
        dv_dt = np.gradient(X[:, 0], T, edge_order=1)
    else:
        dv_dt = np.zeros(len(T))

    cfg = dict(
        sys="vdp",
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        L=L,
        C=C,
        mu=mu,
        v0=v0,
        iL0=iL0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X, dv_dt=dv_dt)
    return cfg, out
