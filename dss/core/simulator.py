"""High-level simulation API.

This module intentionally stays free of visualization dependencies.
Use `apps/streamlit/` for interactive visualization, or consume the returned
`scipy.integrate.OdeResult` directly.

Typical usage
-------------
>>> import numpy as np
>>> from dss.models.pendulum import Pendulum
>>> from dss.core.simulator import simulate
>>> sys = Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.02)
>>> sol, diag = simulate(sys, np.array([0.6, 0.0]), T=10.0, fps=200, return_diagnostics=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from dss.core.solver import Solver


@dataclass(frozen=True)
class SimulationDiagnostics:
    runtime_s: float
    n_points: int
    t_start: float
    t_end: float
    method: str
    rtol: float
    atol: float


def simulate(
    system: Any,
    initial_conditions: Sequence[float] | np.ndarray,
    *,
    T: float = 5.0,
    fps: int = 60,
    t_span: Optional[Tuple[float, float]] = None,
    t_eval: Optional[Sequence[float] | np.ndarray] = None,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-6,
    return_diagnostics: bool = False,
):
    """Run a simulation and return the SciPy solution (and optionally diagnostics)."""
    t0 = perf_counter()
    sol = Solver(
        system=system,
        initial_conditions=np.asarray(initial_conditions, dtype=float),
        T=T,
        fps=fps,
        t_span=t_span,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    ).run()
    dt = perf_counter() - t0

    if not return_diagnostics:
        return sol

    diag = SimulationDiagnostics(
        runtime_s=float(dt),
        n_points=int(sol.t.size),
        t_start=float(sol.t[0]) if sol.t.size else float("nan"),
        t_end=float(sol.t[-1]) if sol.t.size else float("nan"),
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    )
    return sol, diag


class Simulator:
    """Object-oriented wrapper around :func:`simulate`."""

    def __init__(
        self,
        system: Any,
        initial_conditions: Sequence[float] | np.ndarray,
        *,
        T: float = 5.0,
        fps: int = 60,
        method: str = "RK45",
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> None:
        self.system = system
        self.initial_conditions = np.asarray(initial_conditions, dtype=float)
        self.T = float(T)
        self.fps = int(fps)
        self.method = str(method)
        self.rtol = float(rtol)
        self.atol = float(atol)

    def run(self, *, return_diagnostics: bool = False):
        return simulate(
            self.system,
            self.initial_conditions,
            T=self.T,
            fps=self.fps,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            return_diagnostics=return_diagnostics,
        )
