"""
Tests for controllers and the closed-loop system.

Covers:
- LQR linearization produces a stabilising gain matrix
- LQR stabilises a small perturbation near the upright equilibrium
- AutoSwingUp produces bounded, finite forces
- SimpleSwitcher transitions correctly from swing-up to LQR
- ClosedLoopCart integrates without error
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dss.models.inverted_pendulum import InvertedPendulum
from dss.controllers.linearize import linearize_upright
from dss.controllers.lqr_controller import AutoLQR
from dss.controllers.swingup import AutoSwingUp
from dss.controllers.simple_switcher import SimpleSwitcher
from dss.wrappers.closed_loop_cart import ClosedLoopCart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_plant(**kwargs):
    defaults = dict(mode="ideal", length=1.0, mass=0.2, cart_mass=0.5)
    defaults.update(kwargs)
    return InvertedPendulum(**defaults)


def integrate_closed(closed, ic, T=5.0):
    sol = solve_ivp(
        closed.dynamics,
        t_span=(0.0, T),
        y0=ic,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    assert sol.success, f"Integration failed: {sol.message}"
    return sol


# ---------------------------------------------------------------------------
# Linearisation
# ---------------------------------------------------------------------------

class TestLinearization:
    def test_A_shape(self):
        plant = make_plant()
        A, B = linearize_upright(plant)
        assert A.shape == (4, 4)

    def test_B_shape_cart_force_only(self):
        plant = make_plant()
        _, B = linearize_upright(plant, include_pivot_input=False)
        assert B.shape == (4, 1)

    def test_B_shape_with_pivot(self):
        plant = make_plant()
        _, B = linearize_upright(plant, include_pivot_input=True)
        assert B.shape == (4, 2)

    def test_upright_is_unstable(self):
        """At least one eigenvalue of A should have positive real part (upright is unstable)."""
        plant = make_plant()
        A, _ = linearize_upright(plant)
        eigenvalues = np.linalg.eigvals(A)
        assert np.any(eigenvalues.real > 0), "Upright equilibrium must be unstable"

    def test_A_structure(self):
        """A[0,1] = 1 and A[2,3] = 1 (kinematic integrators) regardless of parameters."""
        plant = make_plant()
        A, _ = linearize_upright(plant)
        assert A[0, 1] == pytest.approx(1.0)
        assert A[2, 3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LQR controller
# ---------------------------------------------------------------------------

class TestAutoLQR:
    def test_gain_matrix_shape(self):
        plant = make_plant()
        lqr = AutoLQR(plant)
        assert lqr.K.shape == (1, 4)

    def test_gain_matrix_finite(self):
        plant = make_plant()
        lqr = AutoLQR(plant)
        assert np.all(np.isfinite(lqr.K))

    def test_theta_gain_dominant(self):
        """Angle gain should be large relative to position gain for a typical cart-pole."""
        plant = make_plant()
        lqr = AutoLQR(plant)
        K = lqr.K[0]
        # |K_theta| >> |K_x| for standard Bryson weights
        assert abs(K[2]) > abs(K[0]), "Theta gain should dominate position gain"

    def test_stabilises_near_upright(self):
        """LQR should bring a small perturbation back near the upright within 10 s."""
        plant = make_plant()
        lqr = AutoLQR(plant)
        closed = ClosedLoopCart(system=plant, controller=lqr)

        ic = np.array([0.0, 0.0, np.deg2rad(8.0), 0.0])
        sol = integrate_closed(closed, ic, T=10.0)

        theta_final = abs(sol.y[2, -1])
        assert theta_final < np.deg2rad(2.0), (
            f"LQR did not stabilise: final theta = {np.rad2deg(theta_final):.2f} deg"
        )

    def test_force_is_clipped(self):
        """Output force must not exceed u_max."""
        plant = make_plant()
        u_max = 15.0
        lqr = AutoLQR(plant, u_max=u_max)
        # Large perturbation should saturate
        state = np.array([2.0, 3.0, np.deg2rad(20.0), 5.0])
        u = lqr.cart_force(0.0, state)
        assert abs(u) <= u_max + 1e-9

    def test_at_equilibrium_zero_force(self):
        """Zero force at the exact upright equilibrium."""
        plant = make_plant()
        lqr = AutoLQR(plant)
        u = lqr.cart_force(0.0, np.array([0.0, 0.0, 0.0, 0.0]))
        assert u == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Swing-up controller
# ---------------------------------------------------------------------------

class TestAutoSwingUp:
    def _make(self, **kw):
        plant = make_plant(**kw)
        return plant, AutoSwingUp(plant)

    def test_force_finite(self):
        plant, swing = self._make()
        ic = np.array([0.0, 0.0, np.pi, 0.0])  # hanging down
        u = swing.cart_force(0.0, ic)
        assert np.isfinite(u)

    def test_force_bounded(self):
        plant, swing = self._make()
        for theta in np.linspace(0, np.pi, 20):
            u = swing.cart_force(0.0, np.array([0.0, 0.0, theta, 0.0]))
            assert abs(u) <= swing.F_max + 1e-9

    def test_energy_desired(self):
        """Desired energy should equal 2*m*g*lc (energy at upright from bottom)."""
        plant = make_plant()
        swing = AutoSwingUp(plant)
        expected = 2.0 * plant.m * plant.g * plant.lc
        assert swing.energy_desired() == pytest.approx(expected, rel=1e-9)

    def test_physics_based_ke_default(self):
        """When ke is not specified, default should be 5/(m*lc)."""
        plant = make_plant()
        swing = AutoSwingUp(plant)
        expected_ke = 5.0 / (plant.m * plant.lc)
        assert swing.ke == pytest.approx(expected_ke, rel=1e-9)

    def test_soft_zone_includes_theta_feedback(self):
        """Inside the soft zone, theta perturbation should produce non-zero force."""
        plant = make_plant()
        swing = AutoSwingUp(plant)
        # Exactly at x=0, xdot=0 but small theta near upright
        th = np.deg2rad(3.0)
        state = np.array([0.0, 0.0, th, 0.0])
        u = swing.cart_force(0.0, state)
        # Without theta feedback u would be 0; with it, u should be non-zero
        assert abs(u) > 0.0, "Soft zone should produce force for non-zero theta"


# ---------------------------------------------------------------------------
# SimpleSwitcher
# ---------------------------------------------------------------------------

class TestSimpleSwitcher:
    def _make(self):
        plant = make_plant()
        lqr = AutoLQR(plant)
        swing = AutoSwingUp(plant)
        switch = SimpleSwitcher(plant, lqr_controller=lqr, swingup_controller=swing,
                                engage_angle_deg=20.0, dropout_angle_deg=40.0)
        return plant, lqr, swing, switch

    def test_starts_in_swing_mode(self):
        _, _, _, switch = self._make()
        assert switch.mode == SimpleSwitcher.SWING

    def test_transitions_to_lqr_near_upright(self):
        _, _, _, switch = self._make()
        state = np.array([0.0, 0.0, np.deg2rad(10.0), 0.0])
        switch.cart_force(0.0, state)
        assert switch.mode == SimpleSwitcher.LQR

    def test_stays_swing_far_from_upright(self):
        _, _, _, switch = self._make()
        state = np.array([0.0, 0.0, np.deg2rad(90.0), 0.0])
        switch.cart_force(0.0, state)
        assert switch.mode == SimpleSwitcher.SWING

    def test_dropout_when_diverged(self):
        _, _, _, switch = self._make()
        # Enter LQR
        switch.cart_force(0.0, np.array([0.0, 0.0, np.deg2rad(10.0), 0.0]))
        assert switch.mode == SimpleSwitcher.LQR
        # Now diverge past dropout threshold
        switch.cart_force(1.0, np.array([0.0, 0.0, np.deg2rad(50.0), 0.0]))
        assert switch.mode == SimpleSwitcher.SWING

    def test_output_finite(self):
        plant, _, _, switch = self._make()
        closed = ClosedLoopCart(system=plant, controller=switch)
        for theta in [np.pi, np.pi / 2, np.deg2rad(10.0), 0.0]:
            state = np.array([0.0, 0.0, theta, 0.0])
            out = closed.dynamics(0.0, state)
            assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# ClosedLoopCart integration
# ---------------------------------------------------------------------------

class TestClosedLoopCartIntegration:
    def test_lqr_full_integration(self):
        """Full integration with LQR stabiliser should remain finite and converge."""
        plant = make_plant()
        lqr = AutoLQR(plant)
        closed = ClosedLoopCart(system=plant, controller=lqr)
        ic = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0])
        sol = integrate_closed(closed, ic, T=8.0)
        assert np.all(np.isfinite(sol.y))
        assert abs(sol.y[2, -1]) < np.deg2rad(3.0)

    def test_swingup_lqr_handoff_stabilises(self):
        """Switcher hands off to LQR when theta is already within the engage window
        and LQR then stabilises the pole."""
        plant = make_plant()
        lqr = AutoLQR(plant, u_max=25.0)
        swing = AutoSwingUp(plant, force_limit=25.0)
        switch = SimpleSwitcher(plant, lqr_controller=lqr, swingup_controller=swing,
                                engage_angle_deg=25.0, dropout_angle_deg=45.0)
        closed = ClosedLoopCart(system=plant, controller=switch)

        # Start close to upright so the switcher immediately engages LQR
        ic = np.array([0.0, 0.0, np.deg2rad(20.0), 0.0])
        sol = integrate_closed(closed, ic, T=10.0)
        assert np.all(np.isfinite(sol.y))
        theta_final = abs(sol.y[2, -1])
        assert theta_final < np.deg2rad(2.0), (
            f"Switcher+LQR did not stabilise from 20°: final theta = {np.rad2deg(theta_final):.1f} deg"
        )

    def test_swingup_lqr_integration_stays_finite(self):
        """Full swing-up + LQR integration from hanging down completes without error."""
        plant = make_plant()
        lqr = AutoLQR(plant, u_max=25.0)
        swing = AutoSwingUp(plant, force_limit=25.0)
        switch = SimpleSwitcher(plant, lqr_controller=lqr, swingup_controller=swing,
                                engage_angle_deg=25.0, dropout_angle_deg=45.0)
        closed = ClosedLoopCart(system=plant, controller=switch)

        ic = np.array([0.0, 0.0, np.deg2rad(170.0), 0.0])
        sol = integrate_closed(closed, ic, T=20.0)
        assert np.all(np.isfinite(sol.y))
