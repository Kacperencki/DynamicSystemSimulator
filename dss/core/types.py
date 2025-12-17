from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    """Solver settings expressed in the same terms as the Streamlit UI."""

    t0: float = 0.0
    t1: float = 10.0
    dt: float = 0.01
    method: str = "RK45"
    rtol: float = 1e-4
    atol: float = 1e-6

    def to_solver_kwargs(self) -> Dict[str, Any]:
        T_total = float(self.t1 - self.t0)
        fps_eff = max(1, int(round(1.0 / float(self.dt)))) if self.dt > 0 else 60
        return {
            "T": T_total,
            "fps": fps_eff,
            "t_span": (float(self.t0), float(self.t1)),
            "method": str(self.method),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
        }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    mode: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WrapperConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    initial_state: Sequence[float]
    solver: SolverConfig

    controller: Optional[ControllerConfig] = None
    wrapper: Optional[WrapperConfig] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """Canonical minimum output."""

    T: np.ndarray
    X: np.ndarray

    # Optional extras for dashboards / analysis
    U: Optional[np.ndarray] = None
    meta: Mapping[str, Any] = field(default_factory=dict)
