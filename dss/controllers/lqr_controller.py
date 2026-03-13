from __future__ import annotations

# dss/controllers/lqr_controller.py
"""
Linear-Quadratic Regulator (LQR) for the inverted pendulum / cart-pole.

Theory summary
--------------
LQR minimises the infinite-horizon cost:
    J = ∫ (xᵀQx + uᵀRu) dt

where x = [cart_pos, cart_vel, pole_angle, pole_ang_vel] and u = cart force.

The optimal feedback gain K is found by solving the continuous algebraic
Riccati equation (CARE):  AᵀP + PA − PBR⁻¹BᵀP + Q = 0  →  K = R⁻¹BᵀP

Bryson's rule (used here by default) picks diagonal Q / R entries as:
    Q_ii = 1 / (max_allowed_amplitude_i)²
    R    = 1 / (max_allowed_force)²

Higher Q weight → smaller allowed amplitude → stronger correction of that state.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from typing import Any

from dss.controllers.linearize import linearize_upright
from dss.utils.angles import wrap_to_pi


def brysons_rule_Q(x_max: float, xd_max: float,
                   th_max_rad: float, thd_max: float) -> np.ndarray:
    """Diagonal Q via Bryson's rule: 1/(allowed amplitude)^2."""
    return np.diag(
        [
            1.0 / (float(x_max) ** 2),
            1.0 / (float(xd_max) ** 2),
            1.0 / (float(th_max_rad) ** 2),
            1.0 / (float(thd_max) ** 2),
        ]
    )


def brysons_rule_R(u_max: float) -> np.ndarray:
    """Scalar R via Bryson's rule."""
    return np.array([[1.0 / (float(u_max) ** 2)]], dtype=float)


class AutoLQR:
    """
    Auto-LQR around upright using plant linearization + Bryson defaults.

    State: x = [x, x_dot, theta, theta_dot]^T
    Control: u = cart force

    Control law:
        u = -K · [x - x_ref, x_dot, theta_err - theta_ref, theta_dot]^T

    theta_err is wrapped to (-pi, pi].
    """

    def __init__(
        self,
        system: Any,
        x_max: float = 0.25,
        xd_max: float = 2.0,
        theta_max_deg: float = 8.0,
        thetad_max: float = 4.0,
        u_max: float = 20.0,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        self.system = system

        # Linearization around upright, cart force only
        self.A, self.B = linearize_upright(
            system,
            include_damping=True,
            include_pivot_input=False,
        )

        if Q is None:
            Q = brysons_rule_Q(
                x_max,
                xd_max,
                np.deg2rad(theta_max_deg),
                thetad_max,
            )
        if R is None:
            R = brysons_rule_R(u_max)

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.K = self._solve_lqr(self.A, self.B, self.Q, self.R)  # (1x4)

        self.force_limit = float(u_max)
        self.x_ref = 0.0
        self.theta_ref = 0.0

    @staticmethod
    def _solve_lqr(A: np.ndarray, B: np.ndarray,
                   Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.solve(R, B.T @ P)  # (1x4)

    def cart_force(self, t: float, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = state
        theta_err = wrap_to_pi(float(theta))

        dx = np.array(
            [float(x) - self.x_ref, float(x_dot), theta_err - self.theta_ref, float(theta_dot)],
            dtype=float,
        )

        u = float(-(self.K @ dx.reshape(4, 1))[0, 0])
        return float(np.clip(u, -self.force_limit, self.force_limit))


    # Make controllers uniformly callable: u = pi(t, x)
    def __call__(self, t: float, state: np.ndarray) -> float:
        return self.cart_force(t, state)

    def retune(self, **plant_changes: Any) -> None:
        """
        Optional helper: update plant params then recompute A,B and K.
        e.g., retune(m=0.25, l=0.35)
        """
        for k, v in plant_changes.items():
            setattr(self.system, k, v)

        # recompute derived inertia if needed
        if hasattr(self.system, "Ic") and hasattr(self.system, "lc"):
            self.system.Ip = float(self.system.Ic + self.system.m * (self.system.lc**2))

        self.A, self.B = linearize_upright(
            self.system,
            include_damping=True,
            include_pivot_input=False,
        )
        self.K = self._solve_lqr(self.A, self.B, self.Q, self.R)
