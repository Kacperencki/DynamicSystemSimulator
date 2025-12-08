import numpy as np

class CloseLoopCart:
    """
    Closed-loop wrapper: inverted pendulum plant + any controller
    that exposes cart_force(t, state).
    """
    def __init__(self, system, controller):
        self.system = system
        self.controller = controller

    def dynamics(self, t, state):
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise FloatingPointError(f"[Dynamics Error] Bad state at t={t}: {state}")

        # Controller computes cart force from current state
        u = self.controller.cart_force(t, state)

        if not np.isfinite(u):
            raise FloatingPointError(f"[Control Error] Non-finite control at t={t}: {u}")

        # Treat u as F_cart external; no pivot torque
        return self.system.dynamics(t, state, inputs=u)

    def positions(self, state):
        return self.system.positions(state)

    def energy_check(self, state):
        # delegate to plant
        return self.system.energy_check(state)

    # optional: keep an alias if some code expects .energy()
    def energy(self, state):
        return self.energy_check(state)

    def state_labels(self):
        # let the plant define labels
        return self.system.state_labels()
