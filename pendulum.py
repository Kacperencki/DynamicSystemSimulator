import numpy as np



class Pendulum:
    def __init__(self, length, mass, damping, gravity=9.81):
        self.l = length
        self.m = mass
        self.d = damping
        self.g = gravity



    def dynamics(self, t, y):
        # y to initial conditions???? nie raczej nie y jest z solve_ivp
        theta, dtheta = y # theta = angle, dtheta = angular velocity
        theta_dt = dtheta
        dtheta_dt = - self.d*dtheta/(self.m*self.l*self.l) - self.g / self.l * np.sin(theta)

        return [theta_dt, dtheta_dt]


    def point_coordinates(self, theta):
        x = self.l * np.sin(theta)
        y = - self.l * np.cos(theta)

        return x, y

