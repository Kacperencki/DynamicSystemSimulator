from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np

from dss.core.logger import SimulationLogger
from dss.core.pipeline import run_config, run_system


def maybe_logger(save_run: bool, log_dir: str) -> Optional[SimulationLogger]:
    return SimulationLogger(log_dir=log_dir) if save_run else None


def run_from_cfg(
    cfg: Dict[str, Any],
    *,
    save_run: bool = False,
    log_dir: str = "logs",
    run_name: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run a structured config through the core pipeline.

    Returns:
      - cfg2: the (possibly normalized) structured config returned by the pipeline
      - out : simulation output dict (T, X, and optional extras)
    """
    logger = maybe_logger(save_run, log_dir)
    cfg2, out = run_config(
        cfg,
        logger=logger,
        save_bundle=bool(save_run),
        log_dir=str(log_dir),
        bundle_name=(run_name.strip() or None),
    )
    return cfg2, out


def run_from_system(
    system: Any,
    x0: np.ndarray,
    cfg: Dict[str, Any],
    *,
    save_run: bool = False,
    log_dir: str = "logs",
    run_name: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logger = maybe_logger(save_run, log_dir)
    return run_system(
        system,
        x0,
        cfg=cfg,
        logger=logger,
        save_bundle=bool(save_run),
        log_dir=str(log_dir),
        bundle_name=(run_name.strip() or None),
    )
