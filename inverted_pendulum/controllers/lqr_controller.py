from scipy.linalg import solve_continuous_are
import numpy as np
from DynamicSystemSimulator.inverted_pendulum.linearize import AB_analytic

class LQRController:
    def __init__(self, system, Q=None, R=None, force_limit=30, use_numeric=False):
        self.system = system
        self.Q = np.diag([1.0, 0.1, 200.0, 5.0]) if Q is None else Q # state weight matrix [x, x_dot, theta, theta_dot] importance
        self.R = np.array([[0.5]]) if R is None else R # control weight matrix
        self.F_max = force_limit


    def compute_gain(self):
        A, B = AB_analytic(self.system.M, self.system.m, self.system.l)

        P = solve_continuous_are(A, B, self.Q, self.R) #Solves the continuous-time algebraic Riccati equation
        K = np.linalg.inv(self.R) @ B.T @ P # Matrix multiplication K is [1x4]

        return K



    def __call__(self, t, state):
        current_state_array = np.asarray(state, dtype=float)

        K = self.compute_gain()

        u = float(-K @ current_state_array) # u is the Cart force

        return np.clip(u, -self.F_max, self.F_max)

