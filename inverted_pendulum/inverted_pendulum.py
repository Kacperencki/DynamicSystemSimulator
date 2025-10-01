import numpy as np


class InvertedPendulum:
    def __init__(self, l=0.3, m=0.2, cart_mass=0.5, g=9.81):
        self.l = l
        self.m = m
        self.g = g
        self.M = cart_mass


    def dynamics_ideal(self, t, u, state): # u is the cart force (F)
        x, x_dot, theta, theta_dot = state # x is position of the cart, x_dot is velocity of the cart


        x_double_dot = ((u - self.m * self.g * np.sin(theta) * np.cos(theta) - self.m * self.l * theta_dot**2 * np.sin(theta))
                    / ((self.M + self.m) - self.m * np.cos(theta)**2)) # acceleration of the cart

        theta_double_dot = (self.g * np.sin(theta) - x_double_dot * np.cos(theta)) / self.l


        return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot])

    def positions(self, state):
        x, xdot, theta, thetadot = state
        pivot = np.array([x, 0.0])  # cart’s pivot on the track
        tip = pivot + np.array([self.l * np.sin(theta), self.l * np.cos(theta)])
        return np.vstack([pivot, tip], dtype=float)  # shape (2, 2): [ (x0,y0), (x1,y1) ]

