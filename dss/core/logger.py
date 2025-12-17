# dss/core/logger.py

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


def _to_serializable(obj: Any) -> Any:
    """
    Convert numpy / custom objects to something JSON can handle.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # NumPy scalars
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    # Arrays and sequences
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_to_serializable(x) for x in obj]

    # Dictionaries
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # Fallback: string representation
    return str(obj)


def get_component_metadata(component: Any, role: str) -> Optional[Dict[str, Any]]:
    """
    Extract lightweight metadata for any model/controller/wrapper object.

    Parameters
    ----------
    component : Any
        Object to introspect (model, controller, wrapper, ...).
    role : str
        Human-readable role name, e.g. "system", "controller", "wrapper".

    Returns
    -------
    dict or None
        Dictionary with type and simple numeric / string attributes,
        or None if component is None.
    """
    if component is None:
        return None

    meta: Dict[str, Any] = {
        "role": role,
        "type": f"{component.__class__.__module__}.{component.__class__.__name__}",
    }

    # Collect only simple attributes to avoid dumping huge arrays / objects.
    simple_attrs: Dict[str, Any] = {}
    for name, value in vars(component).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            simple_attrs[name] = value
        elif isinstance(value, (np.floating, np.integer)):
            simple_attrs[name] = value.item()

    meta["params"] = simple_attrs
    return meta


class SimulationLogger:
    """
    Lightweight logger for simulation runs.

    Each call to `log_run` appends one JSON line to `runs.jsonl` in `log_dir`.
    This can be parsed later (e.g. with pandas) to build tables and figures
    for Chapter 6.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.runs_path = os.path.join(self.log_dir, "runs.jsonl")

    def log_run(
        self,
        *,
        system: Any,
        controller: Any = None,
        wrapper: Any = None,
        solver_config: Optional[Dict[str, Any]] = None,
        initial_state: Optional[np.ndarray] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a single simulation run.

        Parameters
        ----------
        system : object
            Dynamical system (model or wrapper) used in the simulation.
        controller : object, optional
            Controller if present.
        wrapper : object, optional
            Wrapper if present (e.g. closed-loop plant+controller).
        solver_config : dict, optional
            Numerical settings such as T, fps, rtol, atol, method, etc.
        initial_state : array-like, optional
            Initial conditions (only shape is logged to keep file small).
        diagnostics : dict, optional
            Aggregated metrics: runtime, n_points, max_energy_error, etc.
        extra_meta : dict, optional
            Free-form metadata: experiment_name, preset name, etc.

        Returns
        -------
        run_id : str
            Unique identifier of the logged run.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        record: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "system": get_component_metadata(system, "system"),
            "controller": get_component_metadata(controller, "controller"),
            "wrapper": get_component_metadata(wrapper, "wrapper"),
            "solver": solver_config or {},
            "initial_state_shape": None,
            "diagnostics": diagnostics or {},
            "extra": extra_meta or {},
        }

        if initial_state is not None:
            arr = np.asarray(initial_state)
            record["initial_state_shape"] = list(arr.shape)

        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_serializable(record)) + "\n")

        return run_id


    def make_run_dir(self, run_id: str) -> str:
        """Create a directory for this run and return its path."""
        run_dir = os.path.join(self.log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_config(self, run_dir: str, config: Dict[str, Any], *, filename: str = "config.json") -> str:
        path = os.path.join(run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(config), f, indent=2)
        return path

    def save_output_npz(
        self,
        run_dir: str,
        *,
        T: np.ndarray,
        X: np.ndarray,
        extras: Optional[Dict[str, Any]] = None,
        filename: str = "output.npz",
    ) -> str:
        path = os.path.join(run_dir, filename)
        payload: Dict[str, Any] = {"T": np.asarray(T), "X": np.asarray(X)}
        if extras:
            for k, v in extras.items():
                if isinstance(v, np.ndarray):
                    payload[k] = v
        np.savez_compressed(path, **payload)
        return path

    def save_bundle(
        self,
        *,
        run_id: str,
        config: Dict[str, Any],
        T: np.ndarray,
        X: np.ndarray,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save config + arrays into a dedicated run directory."""
        run_dir = self.make_run_dir(run_id)
        self.save_config(run_dir, config)
        self.save_output_npz(run_dir, T=T, X=X, extras=extras)
        return run_dir
