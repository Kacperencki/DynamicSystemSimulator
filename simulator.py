import numpy as np
import scipy
from pendulum import Pendulum




class Simulator:
    def __init__(self, system, t_span, theta0, t_eval):
        self.system = system
        self.t_span = t_span
        self.theta0 = theta0
        self.t_eval = t_eval

    def run(self):
        return scipy.integrate.solve_ivp(self.system.dynamics, self.t_span, self.theta0, t_eval=self.t_eval)


