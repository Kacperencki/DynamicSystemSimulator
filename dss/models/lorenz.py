from __future__ import annotations

import numpy as np


class Lorenz:
    """
    Classical Lorenz system (chaotic ODE).

    State:
        x, y, z
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def dynamics(self, t: float, state: np.ndarray,
                 inputs: float | np.ndarray | tuple | None = None) -> np.ndarray:
        x, y, z = state
        x = float(x)
        y = float(y)
        z = float(z)

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return np.array([dx, dy, dz], dtype=float)

    def params(self) -> dict[str, float]:
        return {"sigma": self.sigma, "rho": self.rho, "beta": self.beta}

    def state_labels(self) -> list[str]:
        return ["x", "y", "z"]

    def observable_labels(self) -> list[str]:
        return []

    def observables(self, state: np.ndarray) -> list:
        return []
