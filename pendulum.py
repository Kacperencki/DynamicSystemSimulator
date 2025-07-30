import numpy as np



class Pendulum:
    def __init__(self, length, mass, damping, gravity=9.81):
        self.l = length
        self.m = mass
        self.d = damping
        self.g = gravity

    def dynamics(self, t, state):

        theta, theta_dot = state  # theta = angle, theta_dot = angular velocity
        theta_dt = theta_dot
        dtheta_dt = - self.d*theta_dot/(self.m*self.l*self.l) - self.g / self.l * np.sin(theta)

        return np.array([theta_dt, dtheta_dt])

    def positions(self, state):
        theta, theta_dot = state
        x = self.l * np.sin(theta)
        y = - self.l * np.cos(theta)
        """ 
        Possible change that to include more options for visualizer to take like type of blob to draw on which point
        """
        return [(0, 0), (x, y)]


    def energy_check(self, state):
        theta, theta_dot = state
        x_dot = self.l * theta_dot * np.cos(theta)
        y_dot = self.l * theta_dot * np.sin(theta)

        # Total kinetic energy
        kinetic_energy = self.m * (x_dot**2 + y_dot**2)/2

        # Total potential energy
        h = self.l * (1 - np.cos(theta))
        potential_energy = self.m * self.g * h

        total_energy = kinetic_energy + potential_energy

        return np.array([kinetic_energy, potential_energy, total_energy])