# apps/streamlit/components/dashboards/_common.py
"""
Shared utilities for all dashboard figure builders.

Functions
---------
downsample_idx(n, max_pts)
    Returns an index array of at most max_pts evenly-spaced indices into a
    length-n array.  Used to cap the number of trace points in Plotly charts
    without losing the shape of the signal.

pad_range(y, pad_frac=0.08)
    Computes a (min, max) range with 8% padding — prevents traces from touching
    the plot borders.  Handles constant signals gracefully.

duration_ms_from_frames(T, frame_idx, fps_fallback)
    Converts the simulation time grid to a Plotly frame duration in milliseconds
    so that the animation plays back at real-time speed.

cfg_param(cfg, key, default)
    Safely extracts a model parameter from the nested config dict, with a
    fallback to the flat top-level dict for backwards compatibility.

animation_buttons(frames, duration_ms)
    Returns the Plotly updatemenus list with a ▶⏸ play/pause toggle and a ↩
    reset-to-first-frame button.  Pass redraw=True for 3-D scenes (Lorenz).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def downsample_idx(n: int, max_pts: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if max_pts <= 0 or n <= max_pts:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_pts, dtype=int)


def pad_range(y: np.ndarray, *, pad_frac: float = 0.08) -> Tuple[float, float]:
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    if np.isclose(y0, y1):
        d = 1.0 if np.isclose(y0, 0.0) else abs(y0) * 0.2
        return y0 - d, y1 + d
    d = pad_frac * (y1 - y0)
    return y0 - d, y1 + d


def duration_ms_from_frames(T: np.ndarray, frame_idx: np.ndarray, *, fps_fallback: int = 60) -> int:
    """Compute Plotly animation frame duration from the simulated time grid.

    If the simulation step is coarser than 1/fps, we slow down the animation to
    match simulated time (so 40 s sim takes ~40 s to play). If the simulation is
    finer, we can subsample frames and keep a smooth playback.
    """
    if T is None or frame_idx is None:
        return int(round(1000.0 / max(1, int(fps_fallback))))

    idx = np.asarray(frame_idx, dtype=int)
    if idx.size < 2:
        return int(round(1000.0 / max(1, int(fps_fallback))))

    try:
        dt = float(np.mean(np.diff(np.asarray(T, dtype=float)[idx])))
    except Exception:
        dt = float("nan")

    if not np.isfinite(dt) or dt <= 0:
        return int(round(1000.0 / max(1, int(fps_fallback))))

    return max(1, int(round(1000.0 * dt)))


from typing import Any, Dict, Mapping, Optional


def cfg_params(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model params dict from a structured cfg, or {}."""
    model = cfg.get("model")
    if isinstance(model, dict):
        params = model.get("params")
        if isinstance(params, dict):
            return params
    return {}


def cfg_solver(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the solver dict from a structured cfg, or {}."""
    solver = cfg.get("solver")
    if isinstance(solver, dict):
        return solver
    return {}


def cfg_param(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Get a model parameter from cfg, supporting both nested and flat configs."""
    params = cfg_params(cfg)
    if key in params:
        return params[key]
    # fallback: some legacy pieces may still put parameters at top-level
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def solver_param(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Get a solver parameter from cfg (nested or flat)."""
    solver = cfg_solver(cfg)
    if key in solver:
        return solver[key]
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def animation_buttons(frames, duration_ms: int, *, redraw: bool = False, y: float = 1.08) -> list:
    """Return Plotly updatemenus list: ▶⏸ play/pause toggle + ↩ reset.

    The play/pause button uses Plotly's args2 mechanism: first click plays,
    second click pauses, and so on — a true single-button toggle.

    Usage:
        if frames:
            fig.update_layout(updatemenus=animation_buttons(frames, duration_ms))
    """
    if not frames:
        return []
    first_name = frames[0].name
    return [
        dict(
            type="buttons",
            direction="left",
            showactive=True,
            active=-1,
            x=0.0,
            y=y,
            xanchor="left",
            yanchor="top",
            pad={"r": 4, "t": 4},
            buttons=[
                dict(
                    label="▶⏸",
                    method="animate",
                    # First click: play
                    args=[
                        None,
                        dict(
                            frame=dict(duration=duration_ms, redraw=redraw),
                            fromcurrent=True,
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                    # Second click: pause (alternates with args on each click)
                    args2=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="↩",
                    method="animate",
                    args=[
                        [first_name],
                        dict(
                            frame=dict(duration=0, redraw=redraw),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                ),
            ],
        )
    ]
