"""Angle helpers.

Centralizing these avoids subtle drift between controller implementations.
"""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (radians) to (-pi, pi]."""
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)
