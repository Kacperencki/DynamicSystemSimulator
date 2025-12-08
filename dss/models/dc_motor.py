import numpy as np


class DCMotor:
    def __init__(self, R, L, Ke, Kt, Im, bm,
                 voltage_func=6.0,
                 load_func=0.0):
        self.R = float(R)   # Resistance [ohm]
        self.L = float(L)   # Inductance [H]
        self.Ke = float(Ke) # Back EMF constant [V/(rad/s)]
        self.Kt = float(Kt) # Torque constant [N*m/A]
        self.J  = float(Im) # Rotor inertia [kg*m^2]
        self.bm = float(bm) # Viscous friction [N*m*s/rad]

        # keep old name for backward compatibility if you use it elsewhere
        self.Im = self.J

        self.voltage_func = voltage_func
        self.load_func = load_func

    def dynamics(self, t, state):
        """
        Basic DC motor model:
            L di/dt = v(t) - R*i - Ke*ω
            J dω/dt = Kt*i - bm*ω - τ_load
        """
        i, omega = state

        # allow callables or constants
        V = self.voltage_func(t) if callable(self.voltage_func) else float(self.voltage_func)
        tau_load = self.load_func(t, omega) if callable(self.load_func) else float(self.load_func)

        di_dt = (V - self.R * i - self.back_emf(omega)) / self.L
        domega_dt = (self.torque(i) - self.bm * omega - tau_load) / self.J

        return np.array([di_dt, domega_dt], dtype=float)

    def torque(self, i):
        return self.Kt * i

    def back_emf(self, omega):
        return self.Ke * omega

    def energy_check(self, state):
        i, omega = state
        E_ind = 0.5 * self.L * i**2
        E_kin = 0.5 * self.J * omega**2
        E_tot = E_ind + E_kin
        return np.array([E_ind, E_kin, E_tot])

    def state_labels(self):
        return ["i [A]", "omega [rad/s]"]
