# main.py
from simulator import Simulator
from pendulum import Pendulum
from visualizer import MatplotlibVisualizer

import numpy as np
import matplotlib as plt

def main():
    pendulum = Pendulum(length=2, mass=2, damping=1, ang_position=np.radians(45),ang_velocity=0)
    #theta0 = [np.radians(45), 0]
    sim = Simulator(pendulum)
    result = sim.run()

    vis = MatplotlibVisualizer(pendulum.l)
    for theta in result.y[0]:
        vis.draw_frame(theta)
    vis.show()

if __name__ == "__main__":
    main()
