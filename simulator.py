
from pendulum import Pendulum
from visualizer import MatplotlibVisualizer
from solver import Solver
import numpy as np
import matplotlib as plt


class Simulator:
    def __init__(self, system, initial_conditions):
        self.system = system
        self.initial_cond = initial_conditions
    def simulate(self):
        result = Solver(self.system, self.initial_cond).run()
        vis = MatplotlibVisualizer(self.system, result.y.T) # result.y.T transposes the shape from (n_vars, n_timepoints) → (n_timepoints, n_vars)
        vis.animate()



