import numpy as np
import scipy




class Solver:

    def __init__(self, system, initial_conditions, t_span=None, t_eval=None):
        self.system = system
        self.initial_cond = initial_conditions

        self.t_span = t_span or [0, 30]
        self.t_eval = t_eval or np.linspace(0, 30, 1000)

    def run(self):
        return scipy.integrate.solve_ivp(self.system.dynamics, self.t_span, self.initial_cond, t_eval=self.t_eval)
