from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, TypedDict

import numpy as np


class SimOutput(TypedDict, total=False):
    T: np.ndarray
    X: np.ndarray

    # Optional extras
    U: np.ndarray
    E_parts: Any
    meta: Mapping[str, Any]


def from_ode_result(sol, *, extras: Optional[Mapping[str, Any]] = None) -> SimOutput:
    out: SimOutput = {"T": np.asarray(sol.t), "X": np.asarray(sol.y).T}
    if extras:
        out.update(dict(extras))
    return out


def ensure_minimum(out: MutableMapping[str, Any]) -> None:
    if "T" not in out or "X" not in out:
        raise KeyError("Simulation output must contain keys 'T' and 'X'.")

    T = np.asarray(out["T"])
    X = np.asarray(out["X"])

    if T.ndim != 1:
        raise ValueError(f"'T' must be 1D, got shape {T.shape}.")
    if X.ndim != 2:
        raise ValueError(f"'X' must be 2D (N, n_state), got shape {X.shape}.")
    if X.shape[0] != T.shape[0]:
        raise ValueError(f"Length mismatch: len(T)={T.shape[0]} but X.shape[0]={X.shape[0]}.")
