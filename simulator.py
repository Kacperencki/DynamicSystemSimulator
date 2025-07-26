import numpy as np
import scipy
from pendulum import Pendulum




class Simulator:
    def __init__(self, system, theta0=None, t_span=None, t_eval=None):
        self.system = system
        self.theta0 = theta0

        if theta0 is None:
            self.theta0 = [self.system.ang_pos, self.system.ang_vel]
        else:
            self.theta0 = theta0

        if t_span is None:
            self.t_span = [0, 10]
        else:
            self.t_span = t_span

        if t_eval is None:
            self.t_eval = np.linspace(0, 10, 500)
        else:
            self.t_eval = t_eval

    def run(self):
        return scipy.integrate.solve_ivp(self.system.dynamics, self.t_span, self.theta0, t_eval=self.t_eval)


