import numpy as np


class VanDerPol:
    """Van der Pol oscillator as a parallel LC with a nonlinear current source.

    Nonlinear current source:
        i_nl(v) = μ * (v^3/3 - v)

    States (absolute circuit variables):
        v   : capacitor voltage [V]
        iL  : inductor current   [A]

    Dynamics (KCL at the node, inductor law):
        C * dv/dt + iL + i_nl(v) = 0      ->  dv/dt  = (-iL - i_nl) / C
        L * diL/dt = v                    ->  diL/dt =  v / L
    """

    def __init__(self, L: float = 1.0, C: float = 1.0, mu: float = 1.0) -> None:
        self.L = float(L)
        self.C = float(C)
        self.mu = float(mu)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        v, iL = state

        # Nonlinear current source
        i_nl = self.mu * (v**3 / 3.0 - v)

        dv_dt = (-iL - i_nl) / self.C
        diL_dt = v / self.L

        return np.array([dv_dt, diL_dt], dtype=float)

    def state_labels(self):
        return ["v", "iL"]

    def positions(self, state: np.ndarray):
        # No physical geometry; return phase point for convenience.
        v, iL = state
        return [(float(v), float(iL))]
