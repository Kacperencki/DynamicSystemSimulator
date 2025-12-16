# apps/streamlit/runners/pendulum_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from dss.models.pendulum import Pendulum
from dss.models.double_pendulum import DoublePendulum
from dss.core.solver import Solver


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
) -> Tuple[Dict, Dict]:
    L, m, g = params["L"], params["m"], params["g"]
    mode = params["mode"]
    b, A, w, phi = params["b"], params["A"], params["w"], params["phi"]
    fc = params.get("fc", 0.0)

    theta0, omega0 = ic["theta0"], ic["omega0"]

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    pend = Pendulum(
        length=L,
        mass=m,
        mode=mode,
        damping=b,
        coulomb=fc,
        drive_amplitude=A,
        drive_frequency=w,
        drive_phase=phi,
        gravity=g,
    )

    sol = Solver(
        pend,
        initial_conditions=[theta0, omega0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T
    KE, PE, E = pend.energy_check(X.T)

    cfg = dict(
        sys="single",
        mode=mode,
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        g=g,
        L=L,
        m=m,
        b=b,
        fc=fc,
        A=A,
        w=w,
        phi=phi,
        theta0=theta0,
        omega0=omega0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X, E_parts=(KE, PE, E))
    return cfg, out


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
) -> Tuple[Dict, Dict]:
    l1, m1, l2, m2, g = params["l1"], params["m1"], params["l2"], params["m2"], params["g"]
    mode = params["mode"]
    b1, b2 = params["b1"], params["b2"]
    fc1 = params.get("fc1", 0.0)
    fc2 = params.get("fc2", 0.0)
    A1, w1, phi1 = params["A1"], params["w1"], params["phi1"]
    A2, w2, phi2 = params["A2"], params["w2"], params["phi2"]

    th1_0, w1_0, th2_0, w2_0 = ic["th1_0"], ic["w1_0"], ic["th2_0"], ic["w2_0"]

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    dp = DoublePendulum(
        length1=l1,
        mass1=m1,
        length2=l2,
        mass2=m2,
        mode=mode,
        gravity=g,
        damping1=b1,
        damping2=b2,
        coulomb1=fc1,
        coulomb2=fc2,
        drive1_amplitude=A1,
        drive1_frequency=w1,
        drive1_phase=phi1,
        drive2_amplitude=A2,
        drive2_frequency=w2,
        drive2_phase=phi2,
        mass_model="uniform",
    )

    sol = Solver(
        dp,
        initial_conditions=[th1_0, w1_0, th2_0, w2_0],
        T=T_total,
        fps=fps_eff,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    ).run()
    T = sol.t
    X = sol.y.T

    KE = np.empty_like(T)
    PE = np.empty_like(T)
    E = np.empty_like(T)
    for i, s in enumerate(X):
        k, p, e = dp.energy_check(s)
        KE[i], PE[i], E[i] = k, p, e

    cfg = dict(
        sys="double",
        mode=mode,
        solver_method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        g=g,
        l1=l1,
        m1=m1,
        b1=b1,
        fc1=fc1,
        l2=l2,
        m2=m2,
        b2=b2,
        fc2=fc2,
        A1=A1,
        w1=w1,
        phi1=phi1,
        A2=A2,
        w2=w2,
        phi2=phi2,
        th1_0=th1_0,
        w1_0=w1_0,
        th2_0=th2_0,
        w2_0=w2_0,
        t0=t0,
        t1=t1,
        dt=dt,
    )
    out = dict(T=T, X=X, E_parts=(KE, PE, E))
    return cfg, out
