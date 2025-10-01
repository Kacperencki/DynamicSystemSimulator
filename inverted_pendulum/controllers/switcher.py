import numpy as np


def wrap_to_pi(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi # bounds the theta degree [-np.pi, np.pi] so for example 3pi = 1pi


class Switcher:
    def __init__(self, system, lqr_controller, swingup_controller, theta_on_degree=15, theta_off_degree=20,
                 energy_dot_on=0.2, energy_dot_off=0.4):

        self.s = system
        self.c1 = lqr_controller
        self.c2 = swingup_controller

        self.theta_on = theta_on_degree
        self.theta_off = theta_off_degree

        self.e_dot_on = energy_dot_on
        self.e_dot_off = energy_dot_off

        self.stabilizing = False

    def energy(self, state):
        m = self.s.m
        g = self.s.g
        l = self.s.l

        x, x_dot, theta, theta_dot = state

        return 0.5*m*(l*theta_dot)**2 + m*g*l*(1 - np.cos(theta))


    def energy_desired(self):
        return 2.0 * self.s.m * self.s.g * self.s.l

    def __call__(self, t, state):
        x, x_dot, theta, theta_dot = state

        theta_error = abs(wrap_to_pi(theta))

        energy_dot = abs(self.energy(state) - self.energy_desired())

        #IDK WHAT THAT IS

        energy_dot_normalize = energy_dot / self.energy_desired()

        if self.stabilizing is True:
            if theta_error > self.theta_off or energy_dot_normalize > self.e_dot_off:
                self.stabilizing = False

        else:
            if theta_error < self.theta_on and energy_dot_normalize < self.e_dot_on:
                self.stabilizing = True

        return



