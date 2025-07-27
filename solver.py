import numpy as np
import scipy
from pendulum import Pendulum



class Solver:

    def __init__(self, system, t_span=None, t_eval=None):
        self.system = system

        if t_span is None:
            self.t_span = [0, 10]
        if t_eval is None:
            self.t_eval = np.linspace(0, 10, 500)

    def run(self):
        print(self.system.initial_cond)
        return scipy.integrate.solve_ivp(self.system.dynamics, self.t_span, self.system.initial_cond, t_eval=self.t_eval)
