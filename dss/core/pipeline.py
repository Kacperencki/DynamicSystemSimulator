"""
dss/core/pipeline.py
====================
Config-driven simulation pipeline.

Typical call chain
------------------
  run_config(cfg)
      └─ build_system(cfg)      # constructs model (+ optional controller wrapper)
      └─ run_system(system, x0) # integrates ODE, handles logging, saves bundle

Config dict structure (all keys optional unless noted)
------------------------------------------------------
  {
    "model": {
        "name": "pendulum",   # REQUIRED — registry key (see dss/models/__init__.py)
        "mode": "damped",     # model mode string
        "params": { ... }     # constructor kwargs (UI names accepted, aliases applied)
    },
    "controller": {           # omit for open-loop simulations
        "name": "ip_lqr",
        "params": { ... }
    },
    "wrapper": {              # default "closed_loop_cart" when controller is present
        "name": "closed_loop_cart"
    },
    "initial_state": [0, 0, 0.1, 0],  # REQUIRED in run_config()
    "solver": {
        "t0": 0.0, "t1": 10.0, "dt": 0.01,
        "method": "RK45", "rtol": 1e-4, "atol": 1e-7
    }
  }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dss.core.contracts import DynamicalSystem
from dss.core.solver import Solver
from dss.core.sim_output import SimOutput, from_ode_result, ensure_minimum
from dss.core.logger import SimulationLogger, _to_serializable  # type: ignore
from dss.models import get_model
from dss.controllers import get_controller
from dss.wrappers.closed_loop_cart import ClosedLoopCart


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_solver(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract solver settings from config and convert dt → fps for Solver."""
    solver = cfg.get("solver", {}) or {}
    t0 = float(solver.get("t0", 0.0))
    t1 = float(solver.get("t1", solver.get("T", 10.0)))
    dt = float(solver.get("dt", 0.01))
    method = str(solver.get("method", solver.get("solver_method", "DOP853")))
    rtol = float(solver.get("rtol", 1e-4))
    atol = float(solver.get("atol", 1e-7))

    T_total = float(t1 - t0)
    # Solver uses fps (samples/sec) internally; convert from dt.
    # max(1, ...) guards against dt > 1s edge cases.
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    return dict(
        T=T_total,
        fps=fps_eff,
        t_span=(t0, t1),
        method=method,
        rtol=rtol,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system(cfg: Dict[str, Any]) -> Tuple[DynamicalSystem, Dict[str, Any]]:
    """Build (system, build_meta) from a config dict.

    Supported configurations
    ------------------------
    - model only → open-loop (plant dynamics, no control force)
    - model + controller + wrapper='closed_loop_cart' → closed-loop

    Returns
    -------
    system : DynamicalSystem
        Ready to simulate (open or closed loop).
    build_meta : dict
        Metadata logged alongside the run (model_name, mode, controller name, …).
    """
    # --- resolve model name and mode ---
    model_cfg = cfg.get("model") or {}
    model_name = str(model_cfg.get("name", cfg.get("model_name", ""))).strip()
    if not model_name:
        raise ValueError("Config must include model.name (or model_name).")

    mode = str(model_cfg.get("mode", cfg.get("mode", "default")))
    params = dict(model_cfg.get("params", {}))

    # Avoid passing mode twice (once positionally and once inside params).
    # Also tolerate legacy configs that put "mode" inside params.
    if "mode" in params:
        # If mode was not explicitly provided, prefer the one in params
        if (model_cfg.get("mode") is None) and (cfg.get("mode") is None):
            mode = str(params.pop("mode"))
        else:
            params.pop("mode", None)

    # Build the plant from registry (aliases + kwargs filtering happen inside factory)
    system = get_model(model_name, mode=mode, **params)

    # --- optional controller + wrapper ---
    controller_cfg = cfg.get("controller") or cfg.get("control") or None
    wrapper_cfg = cfg.get("wrapper") or None

    build_meta: Dict[str, Any] = {"model_name": model_name, "mode": mode}

    if controller_cfg:
        ctrl_name = str(controller_cfg.get("name", controller_cfg.get("id", ""))).strip()
        if not ctrl_name:
            raise ValueError("controller.name is required when controller section is present.")
        ctrl_params = dict(controller_cfg.get("params", {}))

        # Some controllers need the plant instance at construction (e.g. AutoLQR linearizes it).
        ctrl_cls = get_controller(ctrl_name)
        try:
            controller = ctrl_cls(system, **ctrl_params)
        except TypeError:
            # Controller doesn't accept the plant as first arg → pass params only.
            controller = ctrl_cls(**ctrl_params)

        # Default wrapper for cart–pole systems
        w_name = str(wrapper_cfg.get("name", "")).strip() if wrapper_cfg else "closed_loop_cart"

        if w_name == "closed_loop_cart":
            # Wrap plant + controller so dynamics() handles control input automatically
            system = ClosedLoopCart(system, controller)
            build_meta["controller"] = ctrl_name
            build_meta["wrapper"] = "closed_loop_cart"
        else:
            raise ValueError(f"Unsupported wrapper '{w_name}'.")

    return system, build_meta


def run_system(
    system: DynamicalSystem,
    x0: np.ndarray,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    logger: Optional[SimulationLogger] = None,
    save_bundle: bool = False,
    log_dir: str = "logs",
    bundle_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], SimOutput]:
    """Integrate an already-constructed system.

    Parameters
    ----------
    system : DynamicalSystem
        Open or closed-loop system with a `.dynamics(t, state)` method.
    x0 : ndarray
        Initial state vector.
    cfg : dict, optional
        Full config dict (used to extract solver settings and for logging).
    logger : SimulationLogger, optional
        If provided, logs run metadata to runs.jsonl.
    save_bundle : bool
        If True, saves config.json + output.npz to `log_dir`.
    """
    cfg = dict(cfg or {})
    solver_kw = _parse_solver(cfg)   # extract t0, t1, dt → fps, method, rtol, atol

    # Run numerical integration via scipy.integrate.solve_ivp (wrapped in Solver)
    sol = Solver(system, initial_conditions=np.asarray(x0, dtype=float), **solver_kw).run()
    # Convert OdeResult → SimOutput TypedDict {T, X, ...}
    out = from_ode_result(sol)
    ensure_minimum(out)   # raises if T/X are empty or mismatched

    # Optional: write run metadata to log file
    if logger is not None:
        logger.log_run(
            system=system,
            controller=getattr(system, "controller", None),
            wrapper=system if hasattr(system, "system") else None,
            solver_config=solver_kw,
            config=cfg,
            result_summary={"N": int(len(out["T"]))},
        )

    if save_bundle:
        _save_bundle(cfg=cfg, out=out, log_dir=log_dir, name=bundle_name)

    return cfg, out


def run_config(
    cfg: Dict[str, Any],
    *,
    logger: Optional[SimulationLogger] = None,
    save_bundle: bool = False,
    log_dir: str = "logs",
    bundle_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], SimOutput]:
    """Build system from config dict and run the full simulation.

    Convenience function that chains build_system() → run_system().
    The config must contain 'initial_state' (or 'x0') and 'model.name'.
    """
    system, build_meta = build_system(cfg)

    # Extract initial state; accept both 'initial_state' and legacy 'x0'
    x0 = np.asarray(cfg.get("initial_state", cfg.get("x0", [])), dtype=float)
    if x0.size == 0:
        raise ValueError("Config must include initial_state (or x0).")

    # Embed build metadata (model name, mode, controller) into config for logging
    cfg2 = dict(cfg)
    cfg2.setdefault("meta", {})
    cfg2["meta"] = dict(cfg2["meta"], **build_meta)

    return run_system(
        system,
        x0,
        cfg=cfg2,
        logger=logger,
        save_bundle=save_bundle,
        log_dir=log_dir,
        bundle_name=bundle_name,
    )


# ---------------------------------------------------------------------------
# Bundle saving (config + arrays → disk)
# ---------------------------------------------------------------------------

def _save_bundle(*, cfg: Dict[str, Any], out: SimOutput, log_dir: str, name: Optional[str]) -> str:
    """Save config.json + output.npz to a uniquely-named subdirectory of log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    run_id = name or cfg.get("run_id") or "run"

    # Avoid overwriting existing runs by appending _1, _2, …
    run_dir = os.path.join(log_dir, str(run_id))
    i = 0
    base = run_dir
    while os.path.exists(run_dir):
        i += 1
        run_dir = f"{base}_{i}"
    os.makedirs(run_dir, exist_ok=True)

    from pathlib import Path
    Path(run_dir, "config.json").write_text(json.dumps(_to_serializable(cfg), indent=2), encoding="utf-8")

    # Save T (time vector) + X (state matrix) + any extra arrays in out
    np.savez_compressed(
        os.path.join(run_dir, "output.npz"),
        T=np.asarray(out["T"]),
        X=np.asarray(out["X"]),
        **{k: np.asarray(v) for k, v in out.items() if k not in ("T", "X") and isinstance(v, np.ndarray)},
    )
    return run_dir
