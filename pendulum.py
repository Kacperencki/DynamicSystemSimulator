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


    def dynamics(self, t, y):
        u, v = y # u = angle, v = angular velocity
        du_dt = v
        dv_dt = self.d*v/(self.m*self.l*self.l) -self.g / self.l * np.sin(u)
        return [du_dt, dv_dt]

