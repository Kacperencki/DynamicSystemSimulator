# dss/wrappers/closed_loop_cart.py
"""
Closed-loop wrapper for the cart-pole (inverted pendulum) system.

What it does
------------
ClosedLoopCart combines a plant (InvertedPendulum) and a controller into a
single object whose .dynamics() method the ODE solver can call directly.

At each solver step:
  1. The controller computes a cart force:  u = π(t, state)
  2. u is passed as the external input to the plant:  ẋ = f(t, x, u)

This means the solver "sees" a single autonomous system even though there is
an inner control loop running at every integration step.

Compatibility
-------------
Delegates positions(), energy_check(), and state_labels() to the underlying
plant so that dashboards and diagnostics work transparently on the wrapper.

Controllers are accepted in two forms:
  - Objects with a .cart_force(t, state) method  (preferred, explicit)
  - Plain callables: u = controller(t, state)    (fallback)
"""

import numpy as np


class ClosedLoopCart:
    """Closed-loop wrapper: inverted pendulum plant + any cart-force controller."""

    def __init__(self, system, controller):
        self.system = system        # InvertedPendulum (or compatible plant)
        self.controller = controller  # AutoLQR, AutoSwingUp, SimpleSwitcher, …

    def dynamics(self, t, state, inputs=None):
        # Guard against NaN/Inf states that would produce meaningless control signals
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise FloatingPointError(f"[Dynamics Error] Bad state at t={t}: {state}")

        # Compute control force — prefer explicit cart_force() if available
        if hasattr(self.controller, "cart_force"):
            u = self.controller.cart_force(t, state)
        else:
            u = self.controller(t, state)

        if not np.isfinite(u):
            raise FloatingPointError(f"[Control Error] Non-finite control at t={t}: {u}")

        # Pass u as cart force (F_ext); no pivot torque from controller
        return self.system.dynamics(t, state, inputs=u)

    # ------------------------------------------------------------------
    # Delegation to plant (transparent for dashboards and diagnostics)
    # ------------------------------------------------------------------

    def positions(self, state):
        """(x, y) positions of key points — delegated to plant."""
        return self.system.positions(state)

    def energy_check(self, state):
        """[T, V, E_total] energy components — delegated to plant."""
        return self.system.energy_check(state)

    def energy(self, state):
        """Alias for energy_check() (some downstream code uses .energy())."""
        return self.energy_check(state)

    def state_labels(self):
        """Human-readable state dimension names — delegated to plant."""
        return self.system.state_labels()
