from simulator import Simulator
from pendulum import Pendulum
from double_pendulum import DoublePendulum
from visualizer import MatplotlibVisualizer
from solver import Solver
import numpy as np
import matplotlib as plt

def main():
    print("Please select anmiation of single (SP) or double pendulum (DP):")
    user_input = input()

    if user_input == "SP":
        pendulum = Pendulum(length=2, mass=2, damping=1)
        initial_conditions = [np.pi/4, 0]
        simul = Simulator(pendulum, initial_conditions)
        simul.simulate()

    elif user_input == "DP":
        d_pendulum = DoublePendulum(length1=2, mass1=4, length2=1, mass2=2)
        initial_conditions = [np.pi/2, 0, np.pi/4, 0]
        simul = Simulator(d_pendulum, initial_conditions)
        simul.simulate()

if __name__ == "__main__":
    main()
