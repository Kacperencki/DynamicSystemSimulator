import numpy
import numpy as np


class DCMotor:
    def __init__(self, R, L, Ke, Kt, Im, bm, voltage_func=6.0, load_func=0.0):
        self.R = float(R) # Resistance [ohm]
        self.L = float(L)  # Inductance [H]
        self.Ke = float(Ke)  # Back EMF constant [V/(rad/s)]
        self.Kt = float(Kt)  # Torque constan [N * m/A]
        self.Im = float(Im)  # Rotor intertia [kg*m^2]
        self.bm = float(bm)  # Viscious friction coeff [N*m*s/rad]

        self.voltage_func = voltage_func
        self.load_func = load_func


    def dynamics(self, t, state):
        """
        Podstawowy model dynamiczny silnika DC:
        L * di/dt = v(t) - R*i - Ke*ω
        J * dω/dt = Kt*i - b*ω - τ_load
        """
        i, omega = state

        V = self.voltage_func
        tau = self.load_func

        di_dt = (V - self.R * i - self.back_emf(omega)) / self.L
        domega_dt = (self.torque(i) - self.bm * omega - tau) / self.I

        return np.array([di_dt, domega_dt])

        # Pomocnicze (jak u Ciebie)

    def torque(self, i):
        """Moment elektromagnetyczny [N·m]"""
        return self.Kt * i

    def back_emf(self, omega):
        """Napięcie SEM [V]"""
        return self.Ke * omega

    def energy_check(self, state):
        """Energia w indukcyjności + energia kinetyczna wirnika"""
        i, omega = state
        E_ind = 0.5 * self.L * i ** 2
        E_kin = 0.5 * self.Im * omega ** 2
        E_tot = E_ind + E_kin
        return np.array([E_ind, E_kin, E_tot])

    def state_labels(self):
        return ["i [A]", "omega [rad/s]"]
