from __future__ import annotations

import numpy as np


class DoublePendulum:
    """
    Two-link pendulum with point masses (massless rods) OR uniform rods.

    Angles are ABSOLUTE from the downward vertical:
        state = [theta1, theta1_dot, theta2, theta2_dot]

    Modes:
      - 'ideal'     : gravity + coupling (no friction, no drives)
      - 'damped'    : gravity + coupling + viscous + Coulomb friction
      - 'driven'    : as 'damped' + harmonic drive(s)
      - 'dc_driven' : as 'damped' + external torque(s) tau_drive

    Mass models:
      - 'point'   : lc = l,   I_com = 0
      - 'uniform' : lc = l/2, I_com = (1/12) m l^2

    Internally, dynamics use relative coords (q1 = theta1, r = theta2 - theta1),
    but the public API is in absolute angles.
    """

    def __init__(self,
                 length1: float, mass1: float,
                 length2: float, mass2: float,
                 mode: str,
                 damping1: float = 0.0, damping2: float = 0.0,
                 coulomb1: float = 0.0, coulomb2: float = 0.0,
                 drive1_amplitude: float = 0.0, drive1_frequency: float = 0.0, drive1_phase: float = 0.0,
                 drive2_amplitude: float = 0.0, drive2_frequency: float = 0.0, drive2_phase: float = 0.0,
                 gravity: float = 9.81,
                 mass_model: str = "uniform",
                 I1: float | None = None, lc1: float | None = None,
                 I2: float | None = None, lc2: float | None = None,
                 coulomb_vel_eps: float = 1e-3,
                 ) -> None:
        # geometry / mass
        self.l1 = float(length1)
        self.m1 = float(mass1)
        self.l2 = float(length2)
        self.m2 = float(mass2)
        self.g  = float(gravity)

        # viscous friction
        self.b1 = float(damping1)
        self.b2 = float(damping2)

        # Coulomb friction magnitudes
        self.fc1 = float(coulomb1)
        self.fc2 = float(coulomb2)
        self.coulomb_vel_eps = float(coulomb_vel_eps)

        # harmonic drives
        self.A1 = float(drive1_amplitude)
        self.f1 = float(drive1_frequency)
        self.phi1 = float(drive1_phase)

        self.A2 = float(drive2_amplitude)
        self.f2 = float(drive2_frequency)
        self.phi2 = float(drive2_phase)

        self.mass_model = mass_model.lower().strip()

        def _inertia_pair(m: float, l: float) -> tuple[float, float]:
            if self.mass_model == "point":
                I = 0.0
                lc = l
            elif self.mass_model == "uniform":
                I = (1.0 / 12.0) * m * (l ** 2)
                lc = 0.5 * l
            else:
                raise ValueError(f"Unknown mass_model: {mass_model!r}. Use 'point' or 'uniform'.")
            return I, lc

        default_I1, default_lc1 = _inertia_pair(self.m1, self.l1)
        default_I2, default_lc2 = _inertia_pair(self.m2, self.l2)

        self.lc1 = float(default_lc1) if lc1 is None else float(lc1)
        self.lc2 = float(default_lc2) if lc2 is None else float(lc2)
        self.I1 = float(default_I1) if I1 is None else float(I1)
        self.I2 = float(default_I2) if I2 is None else float(I2)

        self.mode = str(mode).lower().strip()
        if self.mode not in {"ideal", "damped", "driven", "dc_driven"}:
            raise ValueError(f"Unknown mode: {mode!r}")

    # ======================================================================
    # Public API
    # ======================================================================
    def dynamics(self, t: float, state: np.ndarray,
                 inputs: float | np.ndarray | tuple | None = None,
                 tau_drive: float | tuple | None = None) -> np.ndarray:
        if inputs is not None and tau_drive is None:
            tau_drive = inputs
        if self.mode == "ideal":
            return self._solve_theta_ddot(t, state,
                                          tau1=0.0, tau2=0.0,
                                          use_friction=False,
                                          include_harmonic=False)

        if self.mode == "damped":
            return self._solve_theta_ddot(t, state,
                                          tau1=0.0, tau2=0.0,
                                          use_friction=True,
                                          include_harmonic=False)

        if self.mode == "driven":
            return self._solve_theta_ddot(t, state,
                                          tau1="harmonic1",
                                          tau2="harmonic2",
                                          use_friction=True,
                                          include_harmonic=True)

        if self.mode == "dc_driven":
            # tau_drive can be scalar or (tau1,tau2)
            if tau_drive is None:
                tau1_val, tau2_val = 0.0, 0.0
            elif isinstance(tau_drive, (tuple, list)) and len(tau_drive) == 2:
                tau1_val, tau2_val = float(tau_drive[0]), float(tau_drive[1])
            else:
                tau1_val, tau2_val = float(tau_drive), 0.0

            return self._solve_theta_ddot(t, state,
                                          tau1=tau1_val,
                                          tau2=tau2_val,
                                          use_friction=True,
                                          include_harmonic=False)

        raise RuntimeError(f"Unhandled mode: {self.mode!r}")

    def state_labels(self) -> list[str]:
        return ["theta1", "theta1_dot", "theta2", "theta2_dot"]

    def joint_speed(self, state: np.ndarray) -> float:
        """Speed of joint 1 (for DC motor)."""
        theta1, theta1_dot, _, _ = state
        return float(theta1_dot)

    def positions(self, state: np.ndarray) -> list[tuple[float, float]]:
        """Positions: pivot -> mass1 -> mass2."""
        theta1, _, theta2, _ = state
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)
        return [(0.0, 0.0), (float(x1), float(y1)), (float(x2), float(y2))]

    def energy_check(self, state: np.ndarray) -> np.ndarray:
        """
        Returns [T, V, E] using COMs and rotational inertia.
        Angles are absolute from downward vertical.
        """
        th1, th1d, th2, th2d = state

        # COM positions
        x1 = self.lc1 * np.sin(th1)
        y1 = -self.lc1 * np.cos(th1)
        x2 = self.l1 * np.sin(th1) + self.lc2 * np.sin(th2)
        y2 = -self.l1 * np.cos(th1) - self.lc2 * np.cos(th2)

        # COM velocities
        x1d = self.lc1 * np.cos(th1) * th1d
        y1d = self.lc1 * np.sin(th1) * th1d
        x2d = self.l1 * np.cos(th1) * th1d + self.lc2 * np.cos(th2) * th2d
        y2d = self.l1 * np.sin(th1) * th1d + self.lc2 * np.sin(th2) * th2d

        # kinetic energy
        T_trans = 0.5 * (self.m1 * (x1d ** 2 + y1d ** 2) +
                         self.m2 * (x2d ** 2 + y2d ** 2))
        T_rot = 0.5 * (self.I1 * th1d ** 2 + self.I2 * th2d ** 2)
        T = T_trans + T_rot

        # potential energy (zero at th1=th2=0)
        h1 = self.lc1 * (1.0 - np.cos(th1))
        h2 = self.l1 * (1.0 - np.cos(th1)) + self.lc2 * (1.0 - np.cos(th2))
        V = self.m1 * self.g * h1 + self.m2 * self.g * h2

        return np.array([T, V, T + V], dtype=float)

    # ======================================================================
    # Core M(q)qdd + C(q,qd)qd + G(q) + F_fric = tau
    # ======================================================================
    def _solve_theta_ddot(self, t: float, state: np.ndarray,
                          tau1: float | str,
                          tau2: float | str,
                          use_friction: bool,
                          include_harmonic: bool) -> np.ndarray:
        """
        Build M(q), C(q,qd)qd, G(q), add friction (viscous + Coulomb),
        solve for [theta1_ddot, theta2_ddot].

        tau1, tau2 can be numbers or 'harmonic1'/'harmonic2'.
        """
        theta1, theta1_dot, theta2, theta2_dot = state

        # map torques
        tau1_val = self._map_tau(t, tau1, self.A1, self.f1, self.phi1,
                                 include_harmonic)
        tau2_val = self._map_tau(t, tau2, self.A2, self.f2, self.phi2,
                                 include_harmonic)

        # friction (viscous + Coulomb) in absolute joints
        if use_friction:
            tau1_val -= self._friction_joint(theta1_dot, self.b1, self.fc1)
            tau2_val -= self._friction_joint(theta2_dot, self.b2, self.fc2)

        # relative coords
        q1, r = theta1, (theta2 - theta1)
        q1d, rd = theta1_dot, (theta2_dot - theta1_dot)

        # inertia matrix M(q)
        c = np.cos(r)
        s = np.sin(r)
        m11 = (self.I1 + self.I2
               + self.m1 * (self.lc1 ** 2)
               + self.m2 * (self.l1 ** 2 + self.lc2 ** 2 + 2.0 * self.l1 * self.lc2 * c))
        m12 = self.I2 + self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * c)
        m22 = self.I2 + self.m2 * (self.lc2 ** 2)

        # Coriolis/centrifugal C(q,qd)qd in relative coords
        h = self.m2 * self.l1 * self.lc2 * s
        c1 = -h * (2.0 * q1d * rd + rd * rd)
        c2 = h * (q1d ** 2)

        # gravity terms
        g1 = ((self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.sin(q1)
              + self.m2 * self.lc2 * self.g * np.sin(q1 + r))
        g2 = self.m2 * self.lc2 * self.g * np.sin(q1 + r)

        # RHS
        rhs1 = tau1_val - c1 - g1
        rhs2 = tau2_val - c2 - g2

        # solve 2x2 system
        det = m11 * m22 - m12 * m12
        eps = 1e-9
        if abs(det) < eps:
            det = eps if det >= 0 else -eps

        inv11, inv12, inv22 = m22 / det, -m12 / det, m11 / det

        q1dd = inv11 * rhs1 + inv12 * rhs2
        rdd  = inv12 * rhs1 + inv22 * rhs2

        theta1_ddot = q1dd
        theta2_ddot = q1dd + rdd

        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot],
                        dtype=float)

    def _map_tau(self, t: float, spec: float | str,
                 A: float, f: float, phi: float,
                 include_harmonic: bool) -> float:
        """
        Convert tau spec to a number:
          - if spec is 'harmonicX' and include_harmonic: A*cos(f*t + phi)
          - if spec is numeric: just that value
        """
        if isinstance(spec, (int, float)):
            return float(spec)
        if isinstance(spec, str) and "harmonic" in spec and include_harmonic:
            if A != 0.0 and f != 0.0:
                return A * np.cos(f * t + phi)
        return 0.0

    def _friction_joint(self, qd: float, b: float, fc: float) -> float:
        """
        Combined viscous + Coulomb friction torque for a joint.
        """
        tau_visc = b * qd
        if fc == 0.0:
            tau_coul = 0.0
        else:
            eps = self.coulomb_vel_eps
            if abs(qd) < eps:
                tau_coul = fc * (qd / eps)
            else:
                tau_coul = fc * np.sign(qd)
        return tau_visc + tau_coul
