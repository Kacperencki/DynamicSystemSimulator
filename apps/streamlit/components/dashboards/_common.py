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
