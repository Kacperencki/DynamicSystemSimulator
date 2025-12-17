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


def _parse_solver(cfg: Dict[str, Any]) -> Dict[str, Any]:
    solver = cfg.get("solver", {}) or {}
    t0 = float(solver.get("t0", 0.0))
    t1 = float(solver.get("t1", solver.get("T", 10.0)))
    dt = float(solver.get("dt", 0.01))
    method = str(solver.get("method", solver.get("solver_method", "RK45")))
    rtol = float(solver.get("rtol", 1e-4))
    atol = float(solver.get("atol", 1e-6))

    T_total = float(t1 - t0)
    fps_eff = max(1, int(round(1.0 / dt))) if dt > 0 else 60

    return dict(
        T=T_total,
        fps=fps_eff,
        t_span=(t0, t1),
        method=method,
        rtol=rtol,
        atol=atol,
    )


def build_system(cfg: Dict[str, Any]) -> Tuple[DynamicalSystem, Dict[str, Any]]:
    """Build (system, build_meta) from a config dict.

    Supported:
      - model only (open-loop)
      - model + controller + wrapper='closed_loop_cart'
    """
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

    system = get_model(model_name, mode=mode, **params)

    controller_cfg = cfg.get("controller") or cfg.get("control") or None
    wrapper_cfg = cfg.get("wrapper") or None

    build_meta: Dict[str, Any] = {"model_name": model_name, "mode": mode}

    if controller_cfg:
        ctrl_name = str(controller_cfg.get("name", controller_cfg.get("id", ""))).strip()
        if not ctrl_name:
            raise ValueError("controller.name is required when controller section is present.")
        ctrl_params = dict(controller_cfg.get("params", {}))
        # Some controllers need the plant instance (e.g., LQR)
        ctrl_cls = get_controller(ctrl_name)
        try:
            controller = ctrl_cls(system, **ctrl_params)
        except TypeError:
            controller = ctrl_cls(**ctrl_params)

        if wrapper_cfg:
            w_name = str(wrapper_cfg.get("name", "")).strip()
        else:
            w_name = "closed_loop_cart"

        if w_name == "closed_loop_cart":
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
    """Run integration for an already-constructed system."""
    cfg = dict(cfg or {})
    solver_kw = _parse_solver(cfg)

    sol = Solver(system, initial_conditions=np.asarray(x0, dtype=float), **solver_kw).run()
    out = from_ode_result(sol)
    ensure_minimum(out)

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
    system, build_meta = build_system(cfg)
    x0 = np.asarray(cfg.get("initial_state", cfg.get("x0", [])), dtype=float)
    if x0.size == 0:
        raise ValueError("Config must include initial_state (or x0).")

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


def _save_bundle(*, cfg: Dict[str, Any], out: SimOutput, log_dir: str, name: Optional[str]) -> str:
    os.makedirs(log_dir, exist_ok=True)
    run_id = name or cfg.get("run_id") or "run"
    # ensure unique-ish dir
    run_dir = os.path.join(log_dir, str(run_id))
    i = 0
    base = run_dir
    while os.path.exists(run_dir):
        i += 1
        run_dir = f"{base}_{i}"
    os.makedirs(run_dir, exist_ok=True)

    from pathlib import Path
    Path(run_dir, "config.json").write_text(json.dumps(_to_serializable(cfg), indent=2), encoding="utf-8")

    # save arrays
    np.savez_compressed(
        os.path.join(run_dir, "output.npz"),
        T=np.asarray(out["T"]),
        X=np.asarray(out["X"]),
        **{k: np.asarray(v) for k, v in out.items() if k not in ("T", "X") and isinstance(v, np.ndarray)},
    )
    return run_dir
