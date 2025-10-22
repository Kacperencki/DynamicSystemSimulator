from core.simulator import Simulator
from systems.vanderpoll_circuit import VanDerPol
from systems.double_pendulum import DoublePendulum
from systems.pendulum import Pendulum
import numpy as np

from inverted_pendulum.controllers.lqr_controller import LQRController
from inverted_pendulum.controllers.swingup import EnergySwingUp
from inverted_pendulum.controllers.simple_switcher import SimpleSwitcher
from inverted_pendulum.closed_lood_cart import CloseLoopCart
from inverted_pendulum.inverted_pendulum import InvertedPendulum


def main():

    van = VanDerPol()
    initial_cond = np.array([1.0, 0.0])

    simul = Simulator(system=van, initial_conditions=initial_cond)
    simul.simulate()

    """dp = DoublePendulum(mass1=1, mass2=1, length1=1, length2=1)
    initial_cond = [np.pi/2, 0, np.pi/4, 0]
    simul = Simulator(system=dp, initial_conditions=initial_cond)
    simul.simulate()"""

    """pendulum = Pendulum(length=2.0, mass=2.0, mode="ideal")
    initial_cond = np.array([np.pi/2, 0])

    simul = Simulator(system=pendulum, initial_conditions=initial_cond)
    simul.simulate()"""


    """inv = InvertedPendulum(mode="ideal")
    lqr = LQRController(system=inv)
    swing = EnergySwingUp(system=inv)
    switch = SimpleSwitcher(system=inv ,lqr_controller=lqr, swingup_controller=swing)

    cart = CloseLoopCart(system=inv, controller=switch)

    initial_cond = [0, 0, np.pi, 0]
    simul = Simulator(system=cart, initial_conditions=initial_cond)
    simul.simulate()"""


if __name__ == "__main__":
    main()
