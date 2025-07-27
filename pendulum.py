import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


class Pendulum:
    def __init__(self, length, mass, damping, ang_position, ang_velocity ,gravity=9.81):
        self.l = length
        self.m = mass
        self.d = damping
        self.g = gravity
        self.ang_pos = ang_position
        self.ang_vel = ang_velocity
        self.initial_cond = [self.ang_pos, self.ang_vel]


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

