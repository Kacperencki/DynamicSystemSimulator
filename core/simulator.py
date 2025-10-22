from vizualizer.visualizer import MatplotlibVisualizer
from core.solver import Solver


class Simulator:
    def __init__(self, system, initial_conditions, graph=True):
        self.system = system
        self.initial_cond = initial_conditions
        self.graph = graph
    def simulate(self):
        result = Solver(self.system, self.initial_cond).run()
        print(result)


        vis = MatplotlibVisualizer(self.system, result, graph=self.graph) # result.y.T transposes the shape from (n_vars, n_timepoints) → (n_timepoints, n_vars)
        vis.animate()





