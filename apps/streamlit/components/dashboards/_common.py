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
