# main.py
from simulator import Simulator
from pendulum import Pendulum
from visualizer import MatplotlibVisualizer

import numpy as np
import matplotlib as plt

def main():
    pendulum = Pendulum(length=2, mass=2, damping=1)
    theta0 = [np.radians(45), 0]
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 500)
    sim = Simulator(pendulum, t_span, theta0, t_eval)
    result = sim.run()

    vis = MatplotlibVisualizer(pendulum.l)
    for theta in result.y[0]:
        vis.draw_frame(theta)
    vis.show()

if __name__ == "__main__":
    main()
