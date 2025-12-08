import numpy as np


class Pendulum:
    """
    Single-link pendulum with textbook modes:
      - 'ideal'     : gravity only
      - 'damped'    : gravity + viscous + Coulomb friction
      - 'driven'    : gravity + viscous + Coulomb + harmonic torque A*cos(f*t + phi)
      - 'dc_driven' : gravity + viscous + Coulomb + external torque tau_drive

    State: (theta, theta_dot)
        theta      : angle from downward vertical [rad]
        theta_dot  : angular velocity [rad/s]
    """

    def __init__(self,
                 length,
                 mass,
                 mode,
                 damping=0.0,
                 coulomb=0.0,
                 drive_amplitude=2.0,
                 drive_frequency=2.0,
                 drive_phase=0.0,
                 gravity=9.81,
                 mass_model="point",
                 I=None,
                 lc=None,
                 coulomb_vel_eps=1e-3,
                 ):
        # Basic geometry / parameters
        self.l = float(length)   # [m]
        self.m = float(mass)     # [kg]
        self.b = float(damping)  # viscous friction [N*m*s/rad]
        self.fc = float(coulomb) # Coulomb friction magnitude [N*m]
        self.g = float(gravity)  # [m/s^2]

        # Drive (harmonic)
        self.A = float(drive_amplitude)  # [N*m]
        self.f = float(drive_frequency)  # [rad/s]
        self.phi = float(drive_phase)    # [rad]

        self.mass_model = mass_model.lower().strip()
        default_I, default_lc = self._inertia(self.mass_model, self.m, self.l)

        self.I = float(default_I) if I is None else float(I)      # moment of inertia [kg*m^2]
        self.lc = float(default_lc) if lc is None else float(lc)  # COM distance [m]

        self.mode = str(mode).lower().strip()
        if self.mode not in {"ideal", "damped", "driven", "dc_driven"}:
            raise ValueError(f"Unknown mode: {mode!r}")

        # smoothing for Coulomb friction near zero speed
        self.coulomb_vel_eps = float(coulomb_vel_eps)

    # ======================================================================
    # Public API
    # ======================================================================
    def dynamics(self, t, state, tau_drive=None):
        """
        Main dispatcher. All modes go through the same pattern:
            I*theta_ddot = tau_ext + tau_harm - tau_visc - tau_coul - tau_grav
        """
        if self.mode == "ideal":
            return self._dynamics_core(t, state,
                                       include_viscous=False,
                                       include_coulomb=False,
                                       include_harmonic=False,
                                       tau_ext=0.0)

        if self.mode == "damped":
            return self._dynamics_core(t, state,
                                       include_viscous=True,
                                       include_coulomb=True,
                                       include_harmonic=False,
                                       tau_ext=0.0)

        if self.mode == "driven":
            return self._dynamics_core(t, state,
                                       include_viscous=True,
                                       include_coulomb=True,
                                       include_harmonic=True,
                                       tau_ext=0.0)

        if self.mode == "dc_driven":
            tau_val = 0.0 if tau_drive is None else float(tau_drive)
            return self._dynamics_core(t, state,
                                       include_viscous=True,
                                       include_coulomb=True,
                                       include_harmonic=False,
                                       tau_ext=tau_val)

        raise ValueError(f"Unknown mode: {self.mode}")

    def state_labels(self):
        return ["theta", "theta_dot"]

    def joint_speed(self, state):
        """Joint (pivot) angular speed [rad/s]."""
        _, theta_dot = state
        return float(theta_dot)

    def positions(self, state):
        """Return positions for visualization: pivot -> COM -> tip."""
        theta, _ = state
        x_tip = self.l * np.sin(theta)
        y_tip = -self.l * np.cos(theta)
        x_center = self.lc * np.sin(theta)
        y_center = -self.lc * np.cos(theta)
        return [(0.0, 0.0), (x_center, y_center), (x_tip, y_tip)]

    def energy_check(self, state):
        """Return [T, V, E]."""
        theta, theta_dot = state
        kinetic_energy = 0.5 * self.I * theta_dot**2
        potential_energy = self.m * self.g * self.lc * (1.0 - np.cos(theta))
        total_energy = kinetic_energy + potential_energy
        return np.array([kinetic_energy, potential_energy, total_energy], dtype=float)

    # ======================================================================
    # Core dynamics + friction helpers
    # ======================================================================
    def _dynamics_core(self, t, state,
                       include_viscous,
                       include_coulomb,
                       include_harmonic,
                       tau_ext):
        """
        Unified equation:
            I*theta_ddot = tau_ext + tau_harm - tau_visc - tau_coul - tau_grav
        """
        theta, theta_dot = state

        # external torque (DC motor etc.)
        tau_total = float(tau_ext)

        # harmonic drive
        if include_harmonic and self.A != 0.0 and self.f != 0.0:
            tau_total += self.A * np.cos(self.f * t + self.phi)

        # viscous friction
        if include_viscous and self.b != 0.0:
            tau_total -= self.b * theta_dot

        # Coulomb friction
        if include_coulomb and self.fc != 0.0:
            tau_total -= self._coulomb_tau(theta_dot)

        # gravity
        tau_grav = self.m * self.g * self.lc * np.sin(theta)
        tau_total -= tau_grav

        theta_dt = theta_dot
        theta_ddt = tau_total / self.I
        return np.array([theta_dt, theta_ddt], dtype=float)

    def _coulomb_tau(self, qd):
        """
        Smooth-ish Coulomb friction torque that changes sign with qd.
        For |qd| < eps: scale linearly, for |qd| >= eps: ±fc.
        """
        if self.fc == 0.0:
            return 0.0
        eps = self.coulomb_vel_eps
        if abs(qd) < eps:
            return self.fc * (qd / eps)
        return self.fc * np.sign(qd)

    # ======================================================================
    # Geometry / inertia helper
    # ======================================================================
    def _inertia(self, mass_model, m, l):
        """
        Derive I and lc for simple mass model.
        """
        model = mass_model.lower().strip()
        if model == "point":
            I = m * (l**2)
            lc = l
        elif model == "uniform":
            I = (1.0 / 3.0) * m * (l**2)
            lc = 0.5 * l
        else:
            raise ValueError(f"Unknown mass_model: {mass_model}. Use 'point' or 'uniform'.")
        return I, lc
