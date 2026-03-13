# dss/core/solver.py
"""
Thin wrapper around scipy.integrate.solve_ivp.

To add a new solver method: just pass method="Radau" (or any scipy method) —
no code changes needed here.  Supported: RK45, RK23, DOP853, Radau, BDF, LSODA.

Rule of thumb for tolerances:
  - rtol=1e-4, atol=1e-6  →  fast, good for demos
  - rtol=1e-6, atol=1e-8  →  high accuracy, slower (use for chaos / long runs)
"""

import logging

import numpy as np
import scipy.integrate as integrate


logger = logging.getLogger(__name__)

class Solver:
    """
    Thin wrapper around `scipy.integrate.solve_ivp`.

    Responsibilities:
    - create a time grid based on T and fps (unless custom t_eval is given),
    - call system.dynamics(t, x),
    - perform basic NaN/Inf and success checks.
    """

    def __init__(
        self,
        system,
        initial_conditions,
        T: float = 5.0,
        fps: int = 60,
        t_span=None,
        t_eval=None,
        method: str = "RK45",
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        system : object
            Must expose `dynamics(t, state)`.
        initial_conditions : array-like
            Initial state vector y0.
        T : float
            Final time if t_span is not provided (start is 0).
        fps : int
            Number of output samples per second (if t_eval is not provided).
        t_span : (t0, t1), optional
            Integration interval; default is (0, T).
        t_eval : array-like, optional
            Explicit time grid; overrides T/fps-based grid if given.
        method : str
            Solver method (passed to `solve_ivp`), default "RK45".
        rtol, atol : float
            Relative and absolute tolerances for the integrator.
        """
        self.system = system
        self.initial_cond = np.asarray(initial_conditions, dtype=float)

        self.T = float(T)
        self.fps = int(fps)

        # Time span
        if t_span is None:
            self.t_span = (0.0, self.T)
        else:
            if len(t_span) != 2:
                raise ValueError(f"t_span must have length 2, got {t_span}")
            self.t_span = (float(t_span[0]), float(t_span[1]))

        # Time grid
        if t_eval is None:
            duration = self.t_span[1] - self.t_span[0]
            n_steps = int(np.floor(duration * self.fps)) + 1
            self.t_eval = np.linspace(self.t_span[0], self.t_span[1], n_steps)
        else:
            self.t_eval = np.asarray(t_eval, dtype=float)

        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)

    def run(self):
        """
        Run the numerical integration.

        Returns
        -------
        sol : OdeResult
            Result object from `scipy.integrate.solve_ivp`.
        """
        try:
            sol = integrate.solve_ivp(
                fun=self.system.dynamics,
                t_span=self.t_span,
                y0=self.initial_cond,
                t_eval=self.t_eval,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
        except Exception as e:
            logger.exception("Integration failed")
            raise

        if not sol.success:
            logger.warning("Integration unsuccessful: %s", sol.message)

        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise FloatingPointError(
                "[Solver Error] NaN or Inf detected in simulation result."
            )

        return sol
