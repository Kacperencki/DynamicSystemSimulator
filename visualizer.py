import matplotlib.pyplot as plt
import numpy as np

class MatplotlibVisualizer:
    def __init__(self, length):
        self.l = length

    def draw_frame(self, theta):
        x = self.l * np.sin(theta)
        y = -self.l * np.cos(theta)

        plt.clf()  # clear figure for the next frame
        plt.plot([0, x], [0, y], 'k-', linewidth=2)  # pendulum rod
        plt.plot(x, y, 'ro', markersize=10)  # pendulum bob
        plt.xlim(-self.l - 0.5, self.l + 0.5)
        plt.ylim(-self.l - 0.5, 0.5)
        plt.gca().set_aspect('equal')
        plt.pause(0.01)

    def show(self):
        plt.show()