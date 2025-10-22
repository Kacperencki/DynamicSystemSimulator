import numpy as np
from inverted_pendulum.inverted_pendulum import InvertedPendulum
class CloseLoopCart:
    def __init__(self, system, controller):
        self.system = system
        self.controller = controller

    def dynamics(self, t, state):

        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise FloatingPointError(f"[Dynamics Error] Bad state at t={t}: {state}")

        # Controller computes force from current state (continuous-time control)
        u = self.controller.cart_force(t, state)  # returns a clipped scalar

        if not np.isfinite(u):
            raise FloatingPointError(f"[Control Error] Non-finite control at t={t}: {u}")

        return self.system.dynamics_ideal(t, u, state)
    def positions(self, state):
        return self.system.positions(state)

    def energy_check(self, state):
        return self.system.energy_check(state)

    def state_labels(self):
        return ["x [m]", "x_dot [m/s]", "theta [rad]", "theta_dot [rad/s]"]