"""
dss/core/contracts.py
=====================
Protocol interfaces (structural typing) for the DSS component system.

How it works
------------
Python's `typing.Protocol` enables duck typing with static-checker support.
A class satisfies a protocol if it implements the required methods —
no inheritance needed.  `@runtime_checkable` allows isinstance() checks.

Extension guide
---------------
To add a new dynamical system:
  - Implement `dynamics(t, state, inputs=None) -> np.ndarray`
  - Optionally implement `energy_check`, `positions`, `state_labels`
    to unlock energy plots, animation, and labelled axes respectively.

To add a new controller:
  - Implement `__call__(t, state) -> float`
  - Controllers are plain callables; no base class required.
"""

from __future__ import annotations

from typing import Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

# Type aliases used across the library
Array = np.ndarray
Inputs = Union[float, Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Core contracts (required by the solver)
# ---------------------------------------------------------------------------

@runtime_checkable
class DynamicalSystem(Protocol):
    """Minimal contract required by the numerical solver (Solver / simulate).

    The solver calls system.dynamics(t, state) at each integration step.
    `inputs` carries external forcing (e.g. control force); pass None for open-loop.
    """

    def dynamics(self, t: float, state: Array, inputs: Inputs | None = None) -> Array: ...


@runtime_checkable
class Controller(Protocol):
    """Callable controller: u = π(t, x).

    All controllers in dss/controllers/ implement this via __call__,
    making them drop-in replaceable and composable (e.g. switcher chains two).
    """

    def __call__(self, t: float, state: Array) -> float: ...


# ---------------------------------------------------------------------------
# Optional feature protocols (checked at runtime by dashboards / runners)
# ---------------------------------------------------------------------------

@runtime_checkable
class HasStateLabels(Protocol):
    """System exposes human-readable labels for each state dimension.

    Used by plots and dashboards to label axes automatically.
    Example: ["x [m]", "x_dot [m/s]", "theta [rad]", "theta_dot [rad/s]"]
    """
    def state_labels(self) -> Sequence[str]: ...


@runtime_checkable
class HasPositions(Protocol):
    """System can compute (x, y) positions of key points from a state vector.

    Used by the animation renderer to draw pendulum bobs, cart, etc.
    Returns a list of (x, y) tuples (e.g. [pivot, tip] for a pendulum).
    """
    def positions(self, state: Array) -> Sequence[Tuple[float, float]]: ...


@runtime_checkable
class HasEnergy(Protocol):
    """System can compute its energy components from a state vector.

    Returns an array [T, V, E_total] (kinetic, potential, total).
    Used by runners to compute energy arrays for dashboard energy plots.
    """
    def energy_check(self, state: Array) -> Array: ...
