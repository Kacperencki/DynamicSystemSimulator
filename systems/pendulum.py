import numpy as np



class Pendulum:
    def __init__(self,
                 length,
                 mass,
                 mode,
                 damping=0.0,
                 drive_amplitude=2,
                 drive_frequency=2.0,
                 drive_phase=0.0,
                 gravity=9.81,
                 mass_model="point",
                 I=None,
                 lc=None
                 ):

        # Basic geometry
        self.l = float(length)   # [m]
        self.m = float(mass)     # [kg]
        self.b = float(damping)  # [N*m*s/rad]
        self.g = float(gravity)  # [m/s^2]

        # Drive mode varaibles
        self.A = float(drive_amplitude)  # [N*m]
        self.f = float(drive_frequency)  # [rad/s]
        self.phi = float(drive_phase)    # [rad]

        self.mass_model = mass_model.lower().strip()
        default_I, default_lc = self._inertia(self.mass_model, self.m, self.l)

        self.I = float(default_I) if I is None else float(I)      # I to bezwładność (moment of inertia) [kg*m^2]
        self.lc = float(default_lc) if lc is None else float(lc)  # lc to odległość do środka masy (length to center)

        self.mode = mode


    #Funkcja która za pomocą self.mode uruchamia odpowiednią funkcję pasującą do tego trybu
    def dynamics(self, t, state, tau_drive=None):
        """Dispatcher: routes to the chosen textbook case."""
        if self.mode == "ideal":
            return self.dynamics_ideal(t, state)

        if self.mode == "damped":
            return self.dynamics_damped(t, state)

        if self.mode == "driven":
            return self.dynamics_driven(t, state)

        if self.mode == "dc_driven":
            return self.dynamics_dc_driven(t, state, tau_drive or 0.0)

        raise ValueError(f"Unknown mode: {self.mode}")

    # --- textbook-size ODEs (everything inline; no helper jumping) ---
    def dynamics_ideal(self, t, state):
        """(gravity only)."""
        theta, theta_dot = state
        theta_dt = theta_dot
        dtheta_dt = -(self.m * self.g * self.lc) / self.I * np.sin(theta)
        return np.array([theta_dt, dtheta_dt], dtype=float)

    def dynamics_damped(self, t, state):
        """(viscous + gravity)."""
        theta, theta_dot = state
        tau_damp = self.b * theta_dot
        tau_grav = self.m * self.g * self.lc * np.sin(theta)
        theta_dt = theta_dot
        dtheta_dt = (-tau_damp - tau_grav) / self.I
        return np.array([theta_dt, dtheta_dt], dtype=float)

    def dynamics_driven(self, t, state):
        """(harmonic + damping + gravity)."""
        theta, theta_dot = state
        tau_harm = self.A * np.cos(self.f * t + self.phi) if (self.A and self.f) else 0.0
        tau_damp = self.b * theta_dot
        tau_grav = self.m * self.g * self.lc * np.sin(theta)
        theta_dt = theta_dot
        dtheta_dt = (tau_harm - tau_damp - tau_grav) / self.I
        return np.array([theta_dt, dtheta_dt], dtype=float)

    def dynamics_dc_driven(self, t, state, tau_drive):
        """(external actuator + damping + gravity)."""
        theta, theta_dot = state
        tau_ext = float(tau_drive) if tau_drive is not None else 0.0
        tau_damp = self.b * theta_dot
        tau_grav = self.m * self.g * self.lc * np.sin(theta)
        theta_dt = theta_dot
        dtheta_dt = (tau_ext - tau_damp - tau_grav) / self.I
        return np.array([theta_dt, dtheta_dt], dtype=float)


    """
    optional: unified view for advanced readers (keep commented or separate) ---
    def dynamics_unified(self, t, state, tau_drive=None):
         'I*dθ̇ = τ_ext + A cos(ft+φ) - b θ̇ - m g lc sinθ (terms zeroed per mode).'
         theta, theta_dot = state
         tau_ext  = float(tau_drive) if (self.mode == "dc_driven" and tau_drive is not None) else 0.0
         tau_harm = self.A*np.cos(self.f*t + self.phi) if (self.mode == "driven" and self.A and self.f) else 0.0
         tau_damp = self.b*theta_dot if self.mode != "ideal" else 0.0
         tau_grav = self.m*self.g*self.lc*np.sin(theta)
         
         theta_dt = theta_dot
         dthetha_dt = (tau_ext + tau_harm - tau_damp - tau_grav)/self.I
         return np.array([theta_dt, dtheta_dt], dtype=float)
    """

    def state_labels(self):
        # Name of the state variables (useful for plotting and CSV data storage)
        return ["theta", "theta_dot"]

    def joint_speed(self, state):
        """Prędkość złącza, które napędzasz (pivot)."""
        theta, theta_dot = state
        return float(theta_dot)

    def positions(self, state):
        theta, theta_dot = state
        x_tip = self.l * np.sin(theta)
        y_tip = - self.l * np.cos(theta)
        x_center = self.lc * np.sin(theta)
        y_center = - self.lc * np.cos(theta)

        """ 
        Possible change that to include more options for visualizer to take like type of blob to draw on which point
        """
        return [(0, 0),(x_center, y_center), (x_tip, y_tip)] # Wymaga poprawy w visualizerze

    #Funkcja sprawdza energie kinetyczną i potencjalną
    def energy_check(self, state):
        theta, theta_dot = state

        kinetic_energy = 0.5 * self.I * theta_dot**2
        potential_energy = self.m * self.g * self.lc * (1 - np.cos(theta))

        total_energy = kinetic_energy + potential_energy

        return np.array([kinetic_energy, potential_energy, total_energy], dtype=float)


    # HELPER FUNCTIONS


    def _inertia(self, mass_model, m, l):
        """
        Derive I and lc for simple mass model
        """
        model = mass_model.lower.strip()
        if model == "point":
            I = m * (l**2)
            lc = l
        elif model =="uniform":
            I = (1.0/3.0) * m * (l**2)
            lc = 0.5 * l
        else:
            raise ValueError(f"Unknown mass_model: {mass_model}. Use 'point' or 'uniform'.")

        return I, lc
