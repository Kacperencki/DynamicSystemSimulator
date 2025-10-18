import numpy as np

class DoublePendulum:

    def __init__(self, mass1, length1, mass2, length2, gravity=9.81):
        self.m1 = mass1
        self.m2 = mass2
        self.l1 = length1
        self.l2 = length2
        self.g = gravity

    def dynamics(self, t, state):
        theta1, theta1_dot, theta2, theta2_dot = state  

        delta = theta1 - theta2

        denom = 2*self.m1 + self.m2 - self.m2 * np.cos(2 * delta)

        """theta1_dt = theta1_dot
        theta2_dt = theta2_dot"""

        theta1_double_dot = (
            -self.g * (2 * self.m1 + self.m2) * np.sin(theta1)
            - self.m2 * self.g * np.sin(theta1-2*theta2)
            - 2 * np.sin(delta) * self.m2 * (theta2_dot**2 * self.l2 + theta1_dot**2 * self.l1 * np.cos(delta))
        ) / (self.l1 * denom)

        theta2_double_dot = (
            2 * np.sin(delta) * (
                theta1_dot**2 * self.l1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * np.cos(theta1)
                + theta2_dot**2 * self.l2 * self.m2 * np.cos(delta)
            )
        ) / (self.l2 * denom)

        return np.array([theta1_dot, theta1_double_dot, theta2_dot, theta2_double_dot])

    def positions(self, state):
        theta1, theta1_dot, theta2, theta2_dot = state  # i can get rid of dtheta1 and dtheta2 cause they are velocities
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)

        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)

        return [(0, 0), (x1, y1), (x2, y2)]

    def energy_check(self, state):
        theta1, theta1_dot, theta2, theta2_dot = state

        x1_dot = theta1_dot * self.l1 * np.cos(theta1)
        y1_dot = theta1_dot * self.l1 * np.sin(theta1)

        x2_dot = x1_dot + theta2_dot * self.l2 * np.cos(theta2)
        y2_dot = y1_dot + theta2_dot * self.l2 * np.sin(theta2)

        kinetic_energy = (self.m1 * (x1_dot**2 + y1_dot**2) + self.m2 * (x2_dot**2 + y2_dot**2))/2

        h1 = self.l1 * (1 - np.cos(theta1))
        h2 = h1 + self.l2 * (1 - np.cos(theta2))

        potential_energy = self.m1 * self.g * h1 + self.m2 * self.g * h2

        total_energy = kinetic_energy + potential_energy

        return np.array([kinetic_energy, potential_energy, total_energy])