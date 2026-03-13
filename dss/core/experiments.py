from __future__ import annotations

# dss/core/experiments.py

from typing import Any, Dict, Optional, Sequence, Tuple
from time import perf_counter

import numpy as np

from dss.core.solver import Solver
from dss.core.logger import SimulationLogger


def run_simulation_with_diagnostics(
    system: Any,
    initial_state: Sequence[float] | np.ndarray,
    *,
    T: float = 5.0,
    fps: int = 60,
    t_span: Optional[Tuple[float, float]] = None,
    t_eval: Optional[Sequence[float] | np.ndarray] = None,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-6,
    logger: Optional[SimulationLogger] = None,
    experiment_name: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run a single simulation, compute basic diagnostics and (optionally) log it.

    Parameters
    ----------
    system : object
        Any object exposing `dynamics(t, state)`; may also implement
        optional `energy_check(state)` where `state` is a 1D state vector.
    initial_state : array-like
        Initial conditions for the simulation.
    T, fps, t_span, t_eval, method, rtol, atol :
        Passed to `Solver`.
    logger : SimulationLogger, optional
        If provided, a run summary will be written to `runs.jsonl`.
    experiment_name : str, optional
        Free-form label for this run (e.g. "pendulum_small_angle").
    extra_meta : dict, optional
        Additional metadata (preset name, chapter section, etc.).

    Returns
    -------
    sol : OdeResult
        Raw SciPy solution object from `solve_ivp`.
    diagnostics : dict
        Aggregated metrics: runtime, n_points, max_energy_error, etc.
    """
    solver = Solver(
        system=system,
        initial_conditions=initial_state,
        T=T,
        fps=fps,
        t_span=t_span,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    t0 = perf_counter()
    sol = solver.run()
    runtime = perf_counter() - t0

    diagnostics: Dict[str, Any] = {
        "runtime_s": float(runtime),
        "n_points": int(sol.t.size),
        "t_start": float(sol.t[0]) if sol.t.size > 0 else None,
        "t_end": float(sol.t[-1]) if sol.t.size > 0 else None,
        "rtol": float(rtol),
        "atol": float(atol),
        "method": method,
    }

    # Mechanical energy diagnostics, if available
    if hasattr(system, "energy_check") and sol.t.size > 0:
        try:
            # sol.y has shape (n_states, n_points)
            energies = []
            for k in range(sol.y.shape[1]):
                state_k = sol.y[:, k]          # 1D state at time k
                e_k = system.energy_check(state_k)
                e_k = np.asarray(e_k, dtype=float).ravel()
                energies.append(e_k)

            energies = np.stack(energies, axis=1)  # (n_energy_terms, n_points)
            total_energy = energies[-1, :]
            diagnostics["max_energy_error"] = float(
                np.max(np.abs(total_energy - total_energy[0]))
            )
        except Exception as e:
            diagnostics["energy_check_error"] = str(e)

    # Optional logging
    if logger is not None:
        # If t_span not given, use actual span from solution
        if t_span is not None:
            span_list = [float(t_span[0]), float(t_span[1])]
        elif sol.t.size > 0:
            span_list = [float(sol.t[0]), float(sol.t[-1])]
        else:
            span_list = [0.0, float(T)]

        solver_config = {
            "T": float(T),
            "fps": int(fps),
            "t_span": span_list,
            "method": method,
            "rtol": float(rtol),
            "atol": float(atol),
        }

        meta = dict(extra_meta or {})
        if experiment_name is not None:
            meta["experiment_name"] = experiment_name

        logger.log_run(
            system=system,
            controller=getattr(system, "controller", None),
            wrapper=getattr(system, "wrapper", None),
            solver_config=solver_config,
            initial_state=np.asarray(initial_state, dtype=float),
            diagnostics=diagnostics,
            extra_meta=meta,
        )

    return sol, diagnostics
