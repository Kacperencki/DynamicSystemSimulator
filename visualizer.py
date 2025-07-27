import matplotlib.pyplot as plt
import numpy as np

class MatplotlibVisualizer:
    def __init__(self, system):
        self.system = system

    def pendulum_frame(self, theta): # DO przerobienia
        x, y = self.system.point_coordinates(theta=theta)
        l = self.system.l
        plt.clf()  # clear figure for the next frame
        plt.plot([0, x], [0, y], 'k-', linewidth=2)  # pendulum rod
        plt.plot(x, y, 'ro', markersize=10)  # pendulum bob
        plt.xlim(-l - 0.5, l + 0.5)
        plt.ylim(-l - 0.5, 0.5)
        plt.gca().set_aspect('equal')
        plt.pause(0.01)

    def animation(self, theta_array):
        for theta in theta_array:
            self.pendulum_frame(theta)

        plt.show()
