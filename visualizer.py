import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class MatplotlibVisualizer:
    def __init__(self, system, states):
        self.system = system
        self.states = states

        self.fig, self.ax = plt.subplots()

        # Get number of points to draw from the first state
        first_position = self.system.positions(self.states[0])
        self.num_links = len(first_position) - 1

        # Create empty lines and bobs
        self.lines = []
        self.bobs = []


        for _ in range(self.num_links):
            line, = self.ax.plot([], [], 'k-', linewidth=2)
            self.lines.append(line)

        for _ in range(self.num_links):
            bob, = self.ax.plot([], [], 'ro', markersize=8)
            self.bobs.append(bob)

        # Need to changed to automaticaly being centered around system
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')


    def init_draw(self):

        for line in self.lines:
            line.set_data([], [])
        for bob in self.bobs:
            bob.set_data([], [])

        return self.lines + self.bobs

    def update(self, frame_id):
        state = self.states[frame_id]
        positions = self.system.positions(state)

        # Update lines and bobs between points
        for i, line in enumerate(self.lines):
            x0, y0 = positions[i]
            x1, y1 = positions[i + 1]
            line.set_data([x0, x1], [y0, y1])

        for i, bob in enumerate(self.bobs):
            x, y = positions[i + 1]
            bob.set_data([x], [y])

        return self.lines + self.bobs

    def animate(self):
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.states),
            init_func=self.init_draw,
            blit=False,
            interval=1000 / 60  # 60 FPS
        )
        plt.show()
