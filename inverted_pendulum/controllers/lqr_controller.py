from scipy.linalg import solve_continuous_are
import numpy as np
from inverted_pendulum.linearize import ideal_AB

def wrap_to_pi(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi # bounds the theta degree [-np.pi, np.pi] so for example 3pi = 1pi


"""
    Linear-Quadratic Regulator (LQR) for cart-pole near the upright.

    Workflow:
      1) Build (A,B) for the plant linearized at upright (theta=0).
      2) Solve CARE(A,B,Q,R) -> P, then K = R^{-1} B^T P.
      3) Given state s=[x,xdot,theta,thetadot], compute a wrapped small-angle
         deviation and apply u = -K * (s - s_ref), then clip to actuator limits.

    Notes:
      - The angle used in feedback is 'wrapped' so the linear controller always
        sees a small angle error around 0 (or theta_ref), not e.g. +2π.
      - If the plant parameters change (M,m,l,g), call `compute_gain()` again.
      - Q and R are specified in units of the *state deviations* and input.
    """

class LQRController:
    def __init__(self, system, Q=None, R=None, force_limit=30, use_numeric=False,
                 x_ref=0.0, theta_ref=0.0):

        self.system = system

        self.Q = np.diag([4.0, 0.12, 50.0, 2.0]) if Q is None else Q # state weight matrix [x, x_dot, theta, theta_dot] important
        self.R = np.array([[0.01]]) if R is None else R # control weight matrix

        self.x_ref = x_ref
        self.theta_ref = np.deg2rad(theta_ref)

        self.F_max = force_limit
        self.K = None
        self.compute_gain()

        self.x_ref_target = 0.0  # Where we ultimately want to go (usually 0)
        self.x_ref_tau = 5.0  # Time constant (seconds) for drifting
        self.last_t = None

    def compute_gain(self):
        A, B = ideal_AB(self.system.M, self.system.m, self.system.l, self.system.g)

        P = solve_continuous_are(A, B, self.Q, self.R) #Solves the continuous-time algebraic Riccati equation
        # @ is the operator for multiplying matixes
        self.K = np.linalg.inv(self.R) @ B.T @ P # Matrix multiplication K is [1x4]

    """def update_x_reff(self, t):
        if self.last_t is None:
            self.last_t = t
            return

        delta_t = t - self.last_t

        delta_t = max(1e-6, min(delta_t, 0.1))

        alpha = delta_t / max(self.x_ref_tau, 1e-6)
        self.x_ref = (1 - alpha) * self.x_ref + alpha * self.x_ref_target"""

    def cart_force(self, t, state):
        #self.update_x_reff(t)
        x, x_dot, theta, theta_dot = state

        theta_wrapped = wrap_to_pi(theta=(theta - self.theta_ref))

        current_state_array = np.array([x - self.x_ref, x_dot, theta_wrapped, theta_dot])



        u = float(-self.K @ current_state_array) # u is the Cart force

        return np.clip(u, -self.F_max, self.F_max)

