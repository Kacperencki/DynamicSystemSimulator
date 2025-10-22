import numpy as np


def wrap_to_pi(theta):
    # bounds theta to (-pi, pi]
    return ((theta + np.pi) % (2*np.pi)) - np.pi


class Switcher:
    def __init__(self, system, lqr_controller, swingup_controller,
                 theta_error_on_degree=25, theta_error_off_degree=35,
                 energy_error_on=0.1, energy_error_off=0.2,
                 alpha=0.0, prev_time=None, tau_blend=0.2):

        self.s = system
        self.c_lqr = lqr_controller
        self.c_swing = swingup_controller

        self.theta_error_on = np.deg2rad(theta_error_on_degree)
        self.theta_error_off = np.deg2rad(theta_error_off_degree)

        self.e_error_on = energy_error_on
        self.e_error_off = energy_error_off

        self.alpha = alpha
        self.prev_t = prev_time
        self.tau_blend = tau_blend

        self.just_entered_lqr = False
        self.stabilizing = False  # kept for structure; not used

    def energy(self, state):
        # Bottom-zero total energy: 0 at hanging down (theta=0), 2 m g l at upright (theta=pi)
        m = self.s.m
        g = self.s.g
        l = self.s.l
        x, x_dot, theta, theta_dot = state
        return 0.5 * m * (l * theta_dot)**2 + m * g * l * (1.0 - np.cos(theta))

    def energy_desired(self):
        # Desired energy at upright relative to bottom zero
        return 2.0 * self.s.m * self.s.g * self.s.l

    def smooth_step(self, x, x_on, x_off):
        # Map x in [x_on, x_off] -> [0,1] with smooth (C1) step; clamp outside
        scale = np.clip((x - x_on) / (x_off - x_on + 1e-12), 0.0, 1.0)
        return 3*scale**2 - 2*scale**3

    def readiness(self, theta_error, energy_error):
        # High when both angle and energy are within their "on" windows
        readiness_theta  = 1.0 - self.smooth_step(theta_error,  self.theta_error_on, self.theta_error_off)
        readiness_energy = 1.0 - self.smooth_step(energy_error, self.e_error_on,    self.e_error_off)
        return np.clip(readiness_theta * readiness_energy, 0.0, 1.0)

    def alpha_update(self, t, readiness):
        if self.prev_t is None:
            self.prev_t = t
            self.alpha = float(np.clip(readiness, 0.0, 1.0))
            return self.alpha

        # Compute dt from the current solver time
        delta_t = t - self.prev_t
        self.prev_t = t

        # Defensive clamp in case solver takes a huge internal step
        delta_t = max(1e-6, min(delta_t, 0.1))

        # Exponential smoothing toward readiness with time constant tau_blend
        tau = max(self.tau_blend, 1e-6)
        beta = 1.0 - np.exp(-delta_t / tau)  # in (0,1)
        self.alpha = (1.0 - beta) * self.alpha + beta * np.clip(readiness, 0.0, 1.0)

        # Clamp (numeric safety)
        if self.alpha < 0.0:
            self.alpha = 0.0
        elif self.alpha > 1.0:
            self.alpha = 1.0

        return self.alpha

    def cart_force(self, t, state):
        x, x_dot, theta, theta_dot = state

        # Wrap for angle-based checks
        theta_w = wrap_to_pi(theta)

        # Compute normalized energy error (well-defined with bottom-zero energy)
        E  = self.energy(state)
        Ed = self.energy_desired()
        energy_error = abs(E - Ed) / (Ed + 1e-12)

        # Angle error for readiness
        theta_error = abs(theta_w)

        # Readiness & smoothed blending weight
        readiness = self.readiness(theta_error, energy_error)
        alpha = self.alpha_update(t, readiness)

        # SAFETY: if we're far from upright, do not allow LQR to contribute
        if abs(theta_w) > (np.pi / 2):
            alpha = 0.0
            self.just_entered_lqr = False  # ensure we re-latch x_ref next time

        # Latch x_ref only when alpha is "fully in" (with hysteresis)
        if (alpha > 0.98) and not self.just_entered_lqr:
            self.c_lqr.x_ref = float(x)   # capture cart position for regulation
            self.just_entered_lqr = True
        elif alpha < 0.90:
            self.just_entered_lqr = False

        # GATE LQR CONTRIBUTION: until x_ref is latched, don't blend any LQR
        alpha_eff = alpha if self.just_entered_lqr else 0.0

        # Controller outputs
        u_swing = self.c_swing.cart_force(t, state)
        u_lqr   = self.c_lqr.cart_force(t, state) if alpha_eff > 0.0 else 0.0

        # Blend only with effective alpha
        u = (1.0 - alpha_eff) * u_swing + alpha_eff * u_lqr
        return u
