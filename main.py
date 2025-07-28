from simulator import Simulator
from pendulum import Pendulum
from visualizer import MatplotlibVisualizer
from solver import Solver
import numpy as np
import matplotlib as plt

def main():
    pendulum = Pendulum(length=2, mass=2, damping=1, ang_position=np.pi/4,ang_velocity=0)
    simul = Simulator(pendulum)
    simul.simulate()

if __name__ == "__main__":
    main()
