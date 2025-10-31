import numpy as np

class DoublePendulum:
    """
    Two-link pendulum with point masses (massless rods) OR uniform rods.

    Public angles are ABSOLUTE from the DOWNWARD vertical:
        state = [theta1, theta1_dot, theta2, theta2_dot]

    Modes:
      - 'ideal'     : gravity + nonlinear coupling (no damping, no drives)
      - 'damped'    : + viscous damping at both joints
      - 'driven'    : + harmonic drive (supports joint 1 and/or joint 2)
      - 'dc_driven' : + external torque(s) (tau1 or (tau1, tau2))

    Mass models:
      - 'point'   : lc = l,   I_com = 0
      - 'uniform' : lc = l/2, I_com = (1/12) m l^2

    Internally, dynamics use relative coordinates (q1 = theta1, r = theta2 - theta1)
    for the rigid-body matrices; API remains in absolute angles for clarity.
    """

    def __init__(self,
                 length1, mass1,
                 length2, mass2,
                 mode,
                 damping1=0.0, damping2=0.0,
                 drive1_amplitude=0.0, drive1_frequency=0.0, drive1_phase=0.0,
                 drive2_amplitude=0.0, drive2_frequency=0.0, drive2_phase=0.0,
                 gravity=9.81,
                 mass_model="uniform",
                 I1=None, lc1=None,
                 I2=None, lc2=None
                 ):

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
        self.A1 = float(drive1_amplitude)
        self.f1 = float(drive1_frequency)
        self.phi1 = float(drive1_phase)

        self.A2 = float(drive2_amplitude)
        self.f2 = float(drive2_frequency)
        self.phi2 = float(drive2_phase)

        self.mass_model = mass_model.lower().strip()


        def _inertia_pair(m, l):
            if self.mass_model == "point":
                I = 0.0
                lc = l
                return I, lc

            elif self.mass_model == "uniform":
                I = (0.1/12.0) * m * (l**2)
                lc = 0.5 * l
                return I, lc
            else:
                raise ValueError(f"Unknown mass_model: {mass_model!r}. Use 'point' or 'uniform'.")


        default_I1, default_lc1 = _inertia_pair(self.m1, self.l1)
        default_I2, default_lc2 = _inertia_pair(self.m2, self.l2)

        self.lc1 = float(default_lc1) if lc1 is None else float(lc1)
        self.lc2 = float(default_lc2) if lc2 is None else float(lc2)
        self.I1 = float(default_I1) if I1 is None else float(I1)
        self.I2 = float(default_I2) if I2 is None else float(I2)

        self.mode = str(mode).lower().strip()
        if self.mode not in {"ideal", "damped", "driven", "dc_driven"}:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'ideal', 'damped', 'driven', or 'dc_driven'.")




    # ======================================================================
    # Public API (spójne z single pendulum)
    # ======================================================================

    def dynamics(self, t, state, tau_drive=None):
        if self.mode == "ideal":
            return self.dynamics_ideal(t, state)

        if self.mode == "damped":
            return self.dynamics_damped(t, state)

        if self.mode == "driven":
            return self.dynamics_driven(t, state)

        if self.mode == "dc_driven":
            return self.dynamics_dc_driven(t, state, tau_drive)

        raise RuntimeError(f"Unhandled mode: {self.mode!r}")

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
        return [(0.0, 0.0), (float(x1), float(y1)), (float(x2), float(y2))]

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



        # ------------------------------------------------------------------ #
        # Modes (tiny, readable ODEs)
        # ------------------------------------------------------------------ #
    def dynamics_ideal(self, t, state):
        """Gravity + coupling; no damping, no drives."""
        return self._solve_theta_ddot(t, state,
                                          tau1=0.0, tau2=0.0,
                                          use_damping=False,
                                          include_harmonic=False)

    def dynamics_damped(self, t, state):
        """Add viscous damping at both joints."""
        return self._solve_theta_ddot(t, state,
                                          tau1=0.0, tau2=0.0,
                                          use_damping=True,
                                          include_harmonic=False)

    def dynamics_driven(self, t, state):
        """
        Harmonic drives (either joint can be active):
        tau1 += A1*cos(f1*t + phi1) if A1,f1 set
        tau2 += A2*cos(f2*t + phi2) if A2,f2 set
        """
        return self._solve_theta_ddot(t, state,
                                      tau1="harmonic1",
                                      tau2="harmonic2",
                                      use_damping=True,
                                      include_harmonic=True)

    def dynamics_dc_driven(self, t, state, tau_drive):
        """
        External actuator torques:
            - if scalar: tau1 = scalar, tau2 = 0.0
            - if (tau1, tau2): both joints driven
        """
        tau1_val, tau2_val = 0.0, 0.0
        if tau_drive is None:
            tau1_val, tau2_val = 0.0, 0.0
        elif isinstance(tau_drive, (tuple, list)) and len(tau_drive) == 2:
            tau1_val, tau2_val = float(tau_drive[0]), float(tau_drive[1])
        else:
            tau1_val, tau2_val = float(tau_drive), 0.0

        return self._solve_theta_ddot(t, state,
                                          tau1=tau1_val, tau2=tau2_val,
                                          use_damping=True,
                                          include_harmonic=False)

    def _solve_theta_ddot(self, t, state, tau1, tau2, use_damping, include_harmonic):
        """
        Build M(q), C(q,qd)qd, G(q), then solve for [theta1_ddot, theta2_ddot].
        tau1, tau2 can be numbers or the strings 'harmonic1'/'harmonic2'.
        """
        theta1, theta1_dot, theta2, theta2_dot = state

        # Map torques (compose harmonic and damping in ABSOLUTE joints)
        def harmonic(val, A, f, phi):
            return (A * np.cos(f * t + phi)) if (include_harmonic and A != 0.0 and f != 0.0) else 0.0

        tau1_val = (harmonic(0.0, self.A1, self.f1, self.phi1) if tau1 == "harmonic1"
                    else float(tau1))
        tau2_val = (harmonic(0.0, self.A2, self.f2, self.phi2) if tau2 == "harmonic2"
                    else float(tau2))

        if use_damping:
            tau1_val += -self.b1 * theta1_dot
            tau2_val += -self.b2 * theta2_dot

        # Relative coordinates
        q1, r = theta1, (theta2 - theta1)
        q1d, rd = theta1_dot, (theta2_dot - theta1_dot)

        # Inertia matrix M(q) (with COM inertias)
        c = np.cos(r);
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

        # Gravity
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.sin(q1) + self.m2 * self.lc2 * self.g * np.sin(
            q1 + r)
        g2 = self.m2 * self.lc2 * self.g * np.sin(q1 + r)

        # Right-hand side
        rhs1 = tau1_val - c1 - g1
        rhs2 = tau2_val - c2 - g2

        # Solve 2x2
        det = m11 * m22 - m12 * m12
        eps = 1e-9
        if abs(det) < eps:
            det = eps if det >= 0 else -eps
        inv11, inv12, inv22 = m22 / det, -m12 / det, m11 / det

        q1dd = inv11 * rhs1 + inv12 * rhs2
        rdd = inv12 * rhs1 + inv22 * rhs2

        theta1_ddot = q1dd
        theta2_ddot = q1dd + rdd
        return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot], dtype=float)