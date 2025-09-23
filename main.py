from simulator import Simulator
from pendulum import Pendulum
from double_pendulum import DoublePendulum
from visualizer import MatplotlibVisualizer
from solver import Solver
import numpy as np
import matplotlib as plt

def main():
    """print("Please select anmiation of single (SP) or double pendulum (DP):")
    user_input = input()"""

    print("Damped or Ideal pendulum?: ")
    user_input = input()

    if user_input == "D":
        mode = "damped"
        print("Select mass (2): ")
        mass_input = 2
        print("Select length (2_: ")
        length_input = 2
        print("Select damping (1): ")
        damping_input = 1
        print("Select mass_model (point or uniform):")
        mass_model_input = input()

        pendulum = Pendulum(length=length_input, mass=mass_input, damping=damping_input, mode=mode, mass_model=mass_model_input)
        print("Select initial cond")
        initial_conditions = [np.pi / 4, 0]
        simul2 = Simulator(pendulum, initial_conditions)
        simul2.simulate()
    else:
        print("esle")


    """#if user_input == "SP":
    pendulum = Pendulum(length=2, mass=2, damping=1)
    initial_conditions = [np.pi/4, 0]
    simul = Simulator(pendulum, initial_conditions)
    simul.simulate()"""

    """elif user_input == "DP":
    d_pendulum = DoublePendulum(length1=2, mass1=4, length2=1, mass2=2)
    initial_conditions = [np.pi/2, 0, np.pi/4, 0]
    simul = Simulator(d_pendulum, initial_conditions)
    simul.simulate()"""

if __name__ == "__main__":
    main()
