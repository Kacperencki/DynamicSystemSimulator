import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class MatplotlibVisualizer:

    """
    result = result.y.T z symuylatora
    """
    def __init__(self, system, result, graph):
        self.system = system
        self.time = result.t
        self.result = result.y.T
        self.graph = graph

        self.fig, self.ax = plt.subplots()

        # Get number of points to draw from the first state
        first_position = self.system.positions(self.result[0])
        self.num_links = len(first_position) - 1

        # Create empty lines and bobs
        self.lines = []
        self.bobs = []


        for _ in range(self.num_links):
            line, = self.ax.plot([], [], 'k-', linewidth=2)
            self.lines.append(line)

        for _ in range(self.num_links):
            bob, = self.ax.plot([], [], 'ro', markersize=3)
            self.bobs.append(bob)

        # Need to changed to automaticaly being centered around system
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')


    def init_draw(self):

        for line in self.lines:
            line.set_data([], [])
        for bob in self.bobs:
            bob.set_data([], [])

        return self.lines + self.bobs

    def update(self, frame_id):
        current_state = self.result[frame_id]
        positions = self.system.positions(current_state)

        # Update lines and bobs between points
        for i, line in enumerate(self.lines):
            x0, y0 = positions[i]
            x1, y1 = positions[i + 1]
            line.set_data([x0, x1], [y0, y1])

        for i, bob in enumerate(self.bobs):
            x, y = positions[i + 1]
            bob.set_data([x], [y])

        return self.lines + self.bobs

    def energy_graph(self):

        energies = []
        for state in self.result:
            e = self.system.energy_check(state)
            energies.append(e)
        energies = np.array(energies)

        potential_energy = energies[:, 1]
        kinetic_energy = energies[:, 0]
        total_energy = energies[:, 2]

        fig, ax = plt.subplots()
        time = self.time
        # 3) Plot
        ax.plot(time, kinetic_energy, label="Kinetic energy")
        ax.plot(time, potential_energy, label="Potential energy")
        ax.plot(time, total_energy, label="Energy")
        # 4) Labeling & cosmetics
        ax.set_title("State x vs Time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy [J]")
        ax.grid(True)
        ax.legend(loc="best")

        # 5) Show or save
        plt.show()



    def animate(self, fps=60):


        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.result),
            init_func=self.init_draw,
            blit=True,
            interval=1000 / fps  # 60 FPS
        )
        plt.show()

        print("AAAAAAAAAAAAAAAAAAAAA")
        if self.graph == True:
            self.energy_graph()


