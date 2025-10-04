import numpy as np


def wrap_to_pi(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi # bounds the theta degree [-np.pi, np.pi] so for example 3pi = 1pi


class Switcher:
    def __init__(self, system, lqr_controller, swingup_controller, theta_on_degree=15, theta_off_degree=20,
                 energy_dot_on=0.2, energy_dot_off=0.4, verbose=True):

        self.s = system
        self.c_lqr = lqr_controller
        self.c_swing = swingup_controller

        self.theta_on = np.deg2rad(theta_on_degree)
        self.theta_off = np.deg2rad(theta_off_degree)

        self.e_dot_on = energy_dot_on
        self.e_dot_off = energy_dot_off

        self.stabilizing = False
        self.verbose = verbose

    def energy(self, state):
        m = self.s.m
        g = self.s.g
        l = self.s.l

        x, x_dot, theta, theta_dot = state

        return 0.5*m*(l*theta_dot)**2 + m*g*l*(1 + np.cos(theta))


    def energy_desired(self):
        return 2.0 * self.s.m * self.s.g * self.s.l

    def __call__(self, t, state):
        x, x_dot, theta, theta_dot = state

        theta_error = abs(wrap_to_pi(theta))

        energy = self.energy(state)

        energy_ref = self.energy_desired()

        energy_error = abs(energy-energy_ref)

        #IDK WHAT THAT IS

        energy_normalize = energy_error/energy_ref

        if self.stabilizing is True:
            if theta_error > self.theta_off or energy_normalize > self.e_dot_off:
                self.stabilizing = False


        else:
            if theta_error < self.theta_on and energy_normalize < self.e_dot_on:
                self.stabilizing = True



        return self.c_lqr(t, state) if self.stabilizing is True else self.c_swing(t,state)



