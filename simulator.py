
from pendulum import Pendulum
from visualizer import MatplotlibVisualizer
from solver import Solver
import numpy as np
import matplotlib as plt


class Simulator:
    def __init__(self, system):
        self.system = system

    def simulate(self):
        result = Solver(self.system).run()
        vis = MatplotlibVisualizer(self.system)
        vis.animation(result.y[0]) # y[0] is the theta array needed for calculating point change for pendulum



