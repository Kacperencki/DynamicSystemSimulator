import numpy as np


def wrap_to_pi(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi # bounds the theta degree [-np.pi, np.pi] so for example 3pi = 1pi


class EnergySwingUp:
    def __init__(self, system, ke=8.0, kv=2.0, force_limit=30.0):
        self.system = system
        self.ke = ke
        self.kv = kv
        self.F_max = force_limit

    def energy_desired(self): # energy = 0 is at the bottom [x=0, x_dot=0, thetha=pi, theta_dot=0]
        return 2.0 * self.system.m * self.system.g * self.system.l

    def energy(self, state):
        x, x_dot, theta, theta_dot = state

        return 0.5 * self.system.m * (self.system.l * theta_dot)**2 + self.system.m*self.system.g*self.system.l*(1 + np.cos(theta))

    def __call__(self, t, state):
        x, x_dot, theta, theta_dot = state

        energy_diff = self.energy(state) - self.energy_desired()


        u = self.ke * energy_diff  * theta_dot * np.cos(theta) - self.kv * x_dot

        return np.clip(u, -self.F_max, self.F_max)

