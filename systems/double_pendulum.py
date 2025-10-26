import numpy as np

class DoublePendulum:
    """
    Dwuprzegubowe wahadło z masami punktowymi na bezmasowych prętach.
    Kąty absolutne od pionu: theta1, theta2.
    Stan: [theta1, theta1_dot, theta2, theta2_dot]

    Tryby (spójne z single pendulum):
      - "ideal":      tylko grawitacja i nieliniowe sprzężenia (bez tłumienia, bez napędu)
      - "damped":     + tłumienie lepkie (-b*theta_dot)
      - "driven":     + wymuszenie sinusoidalne (na przegubie 1)
      - "dc_driven":  moment z silnika DC (tau_drive) na przegubie 1
    """

    def __init__(self,
                 length1, mass1,
                 length2, mass2,
                 mode,
                 damping1=0.0, damping2=0.0,
                 drive_amplitude=0.0, drive_frequency=0.0, drive_phase=0.0,
                 gravity=9.81):
        # parametry geometryczne i masowe
        self.l1 = float(length1)
        self.m1 = float(mass1)
        self.l2 = float(length2)
        self.m2 = float(mass2)
        self.g  = float(gravity)

        # tłumienie
        self.b1 = float(damping1)
        self.b2 = float(damping2)

        # parametry wymuszenia sinusoidalnego
        self.a = float(drive_amplitude)
        self.f = float(drive_frequency)
        self.p = float(drive_phase)

        # wybór trybu
        self.mode = mode
        self.mode_map = {
            "ideal": self.dynamics_ideal,
            "damped": self.dynamics_damped,
            "driven": self.dynamics_driven,
            "dc_driven": self.dynamics_dc_driven
        }
        if self.mode not in self.mode_map:
            raise ValueError(f"Unknown mode {self.mode}")

    # ======================================================================
    # Public API (spójne z single pendulum)
    # ======================================================================

    def dynamics(self, t, state, tau_drive=None):
        """Zwraca [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]."""
        return self.mode_map[self.mode](t, state, tau_drive)

    def state_labels(self):
        return ["theta1", "theta1_dot", "theta2", "theta2_dot"]

    def joint_speed(self, state):
        """Prędkość przegubu 1 (potrzebna do obliczeń w silniku DC)."""
        theta1, theta1_dot, theta2, theta2_dot = state
        return float(theta1_dot)

    def positions(self, state):
        """Pozycje: pivot -> mass1 -> mass2 (do wizualizacji)."""
        theta1, _, theta2, _ = state
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)
        return [(0, 0), (x1, y1), (x2, y2)]

    def energy_check(self, state):
        """Zwraca [energia kinetyczna, potencjalna, całkowita]."""
        th1, th1d, th2, th2d = state
        # prędkości mas
        x1d = th1d * self.l1 * np.cos(th1)
        y1d = th1d * self.l1 * np.sin(th1)
        x2d = x1d + th2d * self.l2 * np.cos(th2)
        y2d = y1d + th2d * self.l2 * np.sin(th2)
        kinetic = 0.5 * ( self.m1*(x1d**2 + y1d**2) + self.m2*(x2d**2 + y2d**2) )
        h1 = self.l1 * (1 - np.cos(th1))
        h2 = h1 + self.l2 * (1 - np.cos(th2))
        potential = self.m1*self.g*h1 + self.m2*self.g*h2
        return np.array([kinetic, potential, kinetic + potential])

    # ======================================================================
    # Core mechanics
    # ======================================================================

    def _mass_matrix(self, theta1, theta2):
        """Macierz bezwładności M(q)."""
        delta = theta1 - theta2
        M11 = (self.m1 + self.m2) * self.l1**2
        M22 = self.m2 * self.l2**2
        M12 = self.m2 * self.l1 * self.l2 * np.cos(delta)
        return M11, M12, M22  # macierz symetryczna

    def _passive_acc(self, theta1, theta1_dot, theta2, theta2_dot):
        """Przyspieszenia od grawitacji i nieliniowych sprzężeń (bez momentów zewn.)."""
        delta = theta1 - theta2
        denom = 2*self.m1 + self.m2 - self.m2 * np.cos(2 * delta)

        theta1_ddot_passive = (
            -self.g * (2*self.m1 + self.m2) * np.sin(theta1)
            - self.m2 * self.g * np.sin(theta1 - 2*theta2)
            - 2 * np.sin(delta) * self.m2 * (theta2_dot**2 * self.l2 + theta1_dot**2 * self.l1 * np.cos(delta))
        ) / (self.l1 * denom)

        theta2_ddot_passive = (
            2 * np.sin(delta) * (
                theta1_dot**2 * self.l1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * np.cos(theta1)
                + theta2_dot**2 * self.l2 * self.m2 * np.cos(delta)
            )
        ) / (self.l2 * denom)

        return theta1_ddot_passive, theta2_ddot_passive

    def _active_acc(self, theta1, theta2, torque1, torque2):
        """Zmiana przyspieszeń kątowych spowodowana momentami zewnętrznymi."""
        M11, M12, M22 = self._mass_matrix(theta1, theta2)
        det = M11*M22 - M12**2
        Minv11 =  M22 / det
        Minv12 = -M12 / det
        Minv22 =  M11 / det

        theta1_ddot_active = Minv11 * torque1 + Minv12 * torque2
        theta2_ddot_active = Minv12 * torque1 + Minv22 * torque2
        return theta1_ddot_active, theta2_ddot_active

    # ======================================================================
    # Tryby pracy
    # ======================================================================

    def dynamics_ideal(self, t, state, _unused=None):
        theta1, theta1_dot, theta2, theta2_dot = state
        theta1_ddot, theta2_ddot = self._passive_acc(theta1, theta1_dot, theta2, theta2_dot)

        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])

    def dynamics_damped(self, t, state, _unused=None):
        theta1, theta1_dot, theta2, theta2_dot = state

        theta1_ddot_passive, theta2_ddot_passive = self._passive_acc(
            theta1, theta1_dot, theta2, theta2_dot
        )

        torque1 = -self.b1 * theta1_dot
        torque2 = -self.b2 * theta2_dot

        theta1_ddot_active, theta2_ddot_active = self._active_acc(
            theta1, theta2, torque1, torque2
        )

        theta1_ddot = theta1_ddot_passive + theta1_ddot_active
        theta2_ddot = theta2_ddot_passive + theta2_ddot_active

        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])

    def dynamics_driven(self, t, state, _unused=None):
        theta1, theta1_dot, theta2, theta2_dot = state

        theta1_ddot_passive, theta2_ddot_passive = self._passive_acc(
            theta1, theta1_dot, theta2, theta2_dot
        )

        torque_drive = self.a * np.cos(self.f * t + self.p)
        torque1 = torque_drive - self.b1 * theta1_dot
        torque2 = -self.b2 * theta2_dot

        theta1_ddot_active, theta2_ddot_active = self._active_acc(
            theta1, theta2, torque1, torque2
        )

        theta1_ddot = theta1_ddot_passive + theta1_ddot_active
        theta2_ddot = theta2_ddot_passive + theta2_ddot_active

        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])

    def dynamics_dc_driven(self, t, state, tau_drive):
        """Silnik DC: tau_drive (skalar) działa na przegubie 1."""
        theta1, theta1_dot, theta2, theta2_dot = state

        theta1_ddot_passive, theta2_ddot_passive = self._passive_acc(
            theta1, theta1_dot, theta2, theta2_dot
        )

        torque1 = (0.0 if tau_drive is None else float(tau_drive)) - self.b1 * theta1_dot
        torque2 = -self.b2 * theta2_dot

        theta1_ddot_active, theta2_ddot_active = self._active_acc(
            theta1, theta2, torque1, torque2
        )

        theta1_ddot = theta1_ddot_passive + theta1_ddot_active
        theta2_ddot = theta2_ddot_passive + theta2_ddot_active

        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])
