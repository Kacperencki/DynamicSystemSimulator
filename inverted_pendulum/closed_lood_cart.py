
from DynamicSystemSimulator.inverted_pendulum.cart_controller import CartpoleController
from DynamicSystemSimulator.inverted_pendulum.inverted_pendulum import InvertedPendulum
class CloseLoopCart:
    def __init__(self, system, controller):
        self.system = system
        self.controller = controller

    def dynamics(self, t, state):
        # Controller computes force from current state (continuous-time control)
        u = self.controller.stability(t, state)  # returns a clipped scalar
        return self.system.dynamics_ideal(t, u, state)  # pure plant call

    def positions(self, state):
        return self.system.positions(state)