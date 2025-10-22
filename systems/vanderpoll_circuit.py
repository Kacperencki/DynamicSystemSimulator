import numpy as np

import numpy as np

class VanDerPol:
    """
    Parallel LC with a nonlinear current source i_nl(v) = mu * (v^3/3 - v).
    States: [v, iL] where v is capacitor voltage, iL is inductor current.
    """

    def __init__(self, L=1.0, C=1.0, mu=1.0):
        self.L = float(L)
        self.C = float(C)
        self.mu = float(mu)

    def dynamics(self, t, state):
        v, iL = state
        i_nl = self.mu * ((v**3)/3.0 - v)
        dv_dt = (-iL - i_nl) / self.C
        diL_dt = v / self.L
        return np.array([dv_dt, diL_dt], dtype=float)

    def positions(self, state):
        v, iL = state
        return [(0, 0), (v, iL)]

    def params(self):
        return {"L": self.L, "C": self.C, "mu": self.mu}

    def state_labels(self):
        return ["v [V]", "iL [A]"]

    def observable_labels(self):
        return ["dv_dt [V/s]"]

    def observables(self, state):
        v, iL = state
        i_nl = self.mu * ((v**3)/3.0 - v)
        dv_dt = (-iL - i_nl) / self.C
        return [float(dv_dt)]
