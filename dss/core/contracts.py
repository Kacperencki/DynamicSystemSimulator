from __future__ import annotations

from typing import Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

Array = np.ndarray
Inputs = Union[float, Sequence[float], np.ndarray]


@runtime_checkable
class DynamicalSystem(Protocol):
    """Minimal contract required by the numerical solver."""

    def dynamics(self, t: float, state: Array, inputs: Inputs | None = None) -> Array: ...


@runtime_checkable
class Controller(Protocol):
    """Callable controller contract: u = pi(t, x)."""

    def __call__(self, t: float, state: Array) -> float: ...


@runtime_checkable
class HasStateLabels(Protocol):
    def state_labels(self) -> Sequence[str]: ...


@runtime_checkable
class HasPositions(Protocol):
    def positions(self, state: Array) -> Sequence[Tuple[float, float]]: ...


@runtime_checkable
class HasEnergy(Protocol):
    def energy_check(self, state: Array) -> Array: ...
