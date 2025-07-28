
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
        vis = MatplotlibVisualizer(self.system)
        vis.animation(result.y[0]) # y[0] is the theta array needed for calculating point change for pendulum



