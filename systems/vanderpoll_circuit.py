import numpy as np

class VanDerPol:
    """
    Parallel LC with a nonlinear current source i_nl(v) = mu*(v - v^3/3).
    States: [v, iL] where v is capacitor voltage, iL is inductor current.
    """

    def __init__(self, L=1.0, C=1.0, mu=2.0):
        self.L = float(L)
        self.C = float(C)
        self.mu = float(mu)

    def dynamics(self, t, state):
        """
        Parameters
        ----------
        t : float (unused; for solver compatibility)
        state : array-like [v, iL]

        Returns
        -------
        np.ndarray [dv_dt, diL_dt]
        """
        v, iL = state
        i_nl = -self.mu * (v - (v**3)/3.0)
        dv_dt = (-iL - i_nl) / self.C         # C * dv/dt + iL + i_nl = 0
        diL_dt = v / self.L                    # L * diL/dt = v
        return np.array([dv_dt, diL_dt], dtype=float)

    def positions(self, state):
        """
        Simple plotting helper to mimic your API:
        returns a pair of points (origin) -> (v, iL)
        """
        v, iL = state
        return [(0,0), (v, iL)]


    def params(self):
        return {
            "L": self.L, "C": self.C, "mu": self.mu
        }

    def observables(self, state):

        v, iL = state
        i_nl = -self.mu * (v - (v**3)/3.0)
        dv_dt = (-iL - i_nl) / self.C
        return {"v": float(v), "iL": float(iL), "dv_dt": float(dv_dt)}
