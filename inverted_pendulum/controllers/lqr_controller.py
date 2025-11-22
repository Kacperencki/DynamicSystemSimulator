import numpy as np
from scipy.linalg import solve_continuous_are
from inverted_pendulum.linearize import linearize_upright

def brysons_rule_Q(x_max, xd_max, th_max_rad, thd_max):
    # Diagonal with 1/(allowed amplitude)^2
    return np.diag([1/x_max**2, 1/xd_max**2, 1/th_max_rad**2, 1/thd_max**2])

def brysons_rule_R(u_max):
    return np.array([[1.0 / (u_max**2)]], dtype=float)

class AutoLQR:
    """
    Auto-LQR around upright using plant-aware linearization and Bryson defaults.
    Users can override Q, R or the 'allowed' magnitudes to re-scale sensitivity.
    """

    def __init__(self, system,
                 x_max=0.25, xd_max=2.0,
                 theta_max_deg=8.0, thetad_max=4.0,
                 u_max=20.0,
                 Q=None, R=None):
        self.system = system
        self.A, self.B = linearize_upright(system, include_damping=True, include_pivot_input=False)

        if Q is None:
            Q = brysons_rule_Q(x_max, xd_max, np.deg2rad(theta_max_deg), thetad_max)
        if R is None:
            R = brysons_rule_R(u_max)

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.K = self._solve_lqr(self.A, self.B, self.Q, self.R)

        self.force_limit = float(u_max)
        self.x_ref = 0.0
        self.theta_ref = 0.0

    @staticmethod
    def _solve_lqr(A, B, Q, R):
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ (B.T @ P)  # (1x4)

    def cart_force(self, t, state):
        x, x_dot, theta, theta_dot = state
        theta_err = ((theta + np.pi) % (2*np.pi)) - np.pi  # wrap
        dx = np.array([x - self.x_ref, x_dot, theta_err - self.theta_ref, theta_dot], dtype=float)
        u = float(-self.K @ dx)
        return float(np.clip(u, -self.force_limit, self.force_limit))

    def retune(self, **plant_changes):
        """
        Optional helper: update plant params then recompute A,B and K.
        e.g., retune(m=0.25, l=0.35)
        """
        for k, v in plant_changes.items():
            setattr(self.system, k, v)
        # recompute derived inertia if needed
        if hasattr(self.system, "Ic") and hasattr(self.system, "lc"):
            self.system.Ip = float(self.system.Ic + self.system.m * (self.system.lc**2))
        self.A, self.B = linearize_upright(self.system, include_damping=True, include_pivot_input=False)
        self.K = self._solve_lqr(self.A, self.B, self.Q, self.R)
