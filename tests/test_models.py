"""
Tests for all system models.

Covers:
- Energy conservation in ideal (frictionless, unforced) mode
- Smoke test: dynamics() returns correct shape and finite values
- state_labels() and positions() return expected shapes
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dss.models.pendulum import Pendulum
from dss.models.double_pendulum import DoublePendulum
from dss.models.inverted_pendulum import InvertedPendulum
from dss.models.lorenz import Lorenz
from dss.models.dc_motor import DCMotor
from dss.models.vanderpol_circuit import VanDerPol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def integrate(system, ic, T=5.0):
    """Integrate system.dynamics over [0, T] and return trajectory."""
    sol = solve_ivp(
        system.dynamics,
        t_span=(0.0, T),
        y0=ic,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    assert sol.success, f"Integration failed: {sol.message}"
    return sol


def energy_drift_fraction(system, sol):
    """Max fractional energy drift over the trajectory (relative to initial energy)."""
    energies = np.array([system.energy_check(sol.y[:, i]) for i in range(sol.y.shape[1])])
    total = energies[:, 2]  # E = T + V column
    E0 = abs(total[0])
    if E0 < 1e-12:
        return np.max(np.abs(total))
    return np.max(np.abs(total - total[0])) / E0


# ---------------------------------------------------------------------------
# Pendulum
# ---------------------------------------------------------------------------

class TestPendulum:
    def _sys(self):
        return Pendulum(length=1.0, mass=1.0, mode="ideal")

    def test_dynamics_shape(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([0.3, 0.0]))
        assert np.asarray(out).shape == (2,)

    def test_dynamics_finite(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([1.0, 2.0]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 2

    def test_positions_shape(self):
        positions = self._sys().positions(np.array([0.5, 0.0]))
        assert len(positions) == 3  # origin, COM, tip

    def test_energy_conservation_ideal(self):
        sys = self._sys()
        ic = np.array([np.deg2rad(30.0), 0.0])
        sol = integrate(sys, ic, T=10.0)
        drift = energy_drift_fraction(sys, sol)
        assert drift < 1e-4, f"Energy drift too large: {drift:.2e}"

    def test_small_angle_period(self):
        """For small angles, period ≈ 2π√(l/g). Measure via three zero crossings (one full period)."""
        l, g = 1.0, 9.81
        sys = Pendulum(length=l, mass=1.0, mode="ideal", gravity=g)
        ic = np.array([np.deg2rad(5.0), 0.0])
        T_expected = 2 * np.pi * np.sqrt(l / g)
        # Use a fine t_eval grid so zero-crossing detection is accurate
        t_eval = np.linspace(0.0, T_expected * 4, 4000)
        sol = solve_ivp(sys.dynamics, (0.0, T_expected * 4), ic,
                        method="DOP853", rtol=1e-8, atol=1e-10, t_eval=t_eval)
        theta = sol.y[0]
        # Three consecutive zero crossings span one full period
        crossings = np.where(np.diff(np.sign(theta)))[0]
        assert len(crossings) >= 3, "Not enough oscillation cycles detected"
        T_measured = sol.t[crossings[2]] - sol.t[crossings[0]]
        assert abs(T_measured - T_expected) / T_expected < 0.02

    def test_damped_energy_decreases(self):
        sys = Pendulum(length=1.0, mass=1.0, mode="damped", damping=0.5)
        ic = np.array([np.deg2rad(45.0), 0.0])
        sol = integrate(sys, ic, T=10.0)
        energies = np.array([sys.energy_check(sol.y[:, i])[2] for i in range(sol.y.shape[1])])
        assert energies[-1] < energies[0], "Energy should decrease with damping"


# ---------------------------------------------------------------------------
# Double pendulum
# ---------------------------------------------------------------------------

class TestDoublePendulum:
    def _sys(self):
        return DoublePendulum(length1=1.0, mass1=1.0, length2=1.0, mass2=1.0, mode="ideal")

    def test_dynamics_shape(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([0.3, 0.0, 0.2, 0.0]))
        assert np.asarray(out).shape == (4,)

    def test_dynamics_finite(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([0.5, 1.0, -0.5, 0.5]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 4

    def test_energy_conservation_ideal(self):
        sys = self._sys()
        ic = np.array([np.deg2rad(20.0), 0.0, np.deg2rad(10.0), 0.0])
        sol = integrate(sys, ic, T=5.0)
        drift = energy_drift_fraction(sys, sol)
        assert drift < 1e-3, f"Energy drift too large: {drift:.2e}"


# ---------------------------------------------------------------------------
# Inverted pendulum
# ---------------------------------------------------------------------------

class TestInvertedPendulum:
    def _sys(self, mode="ideal"):
        return InvertedPendulum(mode=mode, length=1.0, mass=0.2, cart_mass=0.5)

    def test_dynamics_shape(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([0.0, 0.0, 0.1, 0.0]))
        assert np.asarray(out).shape == (4,)

    def test_dynamics_finite(self):
        sys = self._sys()
        out = sys.dynamics(0.0, np.array([0.1, 0.5, 0.2, -0.3]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 4

    def test_positions_shape(self):
        positions = self._sys().positions(np.array([0.0, 0.0, 0.1, 0.0]))
        assert len(positions) == 2  # pivot and tip

    def test_energy_conservation_ideal(self):
        """Energy is conserved in ideal mode (no friction, no inputs)."""
        sys = self._sys(mode="ideal")
        ic = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0])
        sol = integrate(sys, ic, T=3.0)
        drift = energy_drift_fraction(sys, sol)
        assert drift < 1e-3, f"Energy drift too large: {drift:.2e}"

    def test_upright_is_equilibrium(self):
        """State derivative is zero at the upright equilibrium with no inputs."""
        sys = self._sys(mode="ideal")
        state = np.array([0.0, 0.0, 0.0, 0.0])
        dstate = sys.dynamics(0.0, state)
        assert np.allclose(dstate, 0.0, atol=1e-12)

    def test_modes_all_run(self):
        """All declared modes execute without error."""
        modes = ["ideal", "damped_cart", "damped_pend", "damped_both", "driven", "dc_driven"]
        ic = np.array([0.0, 0.0, 0.1, 0.0])
        for mode in modes:
            sys = InvertedPendulum(
                mode=mode, length=1.0, mass=0.2, cart_mass=0.5,
                b_cart=0.1, b_pend=0.05,
                cart_drive_amp=1.0, cart_drive_freq=2.0,
            )
            out = sys.dynamics(1.0, ic)
            assert np.all(np.isfinite(out)), f"Non-finite output for mode={mode}"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            InvertedPendulum(mode="bogus")

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            InvertedPendulum(length=-1.0)

    def test_gravity_coupling_sign(self):
        """Small positive theta should produce positive theta_ddot (upright is unstable)."""
        sys = self._sys(mode="ideal")
        ic = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0])
        dstate = sys.dynamics(0.0, ic)
        theta_ddot = dstate[3]
        assert theta_ddot > 0, "theta_ddot should be positive for theta>0 (unstable upright)"

    def test_damped_energy_decreases(self):
        sys = self._sys(mode="damped_both")
        sys.b_cart = 0.5
        sys.b_pend = 0.1
        ic = np.array([0.0, 0.5, np.deg2rad(5.0), 0.5])
        sol = integrate(sys, ic, T=5.0)
        energies = np.array([sys.energy_check(sol.y[:, i])[2] for i in range(sol.y.shape[1])])
        assert energies[-1] < energies[0], "Energy should decrease with damping"


# ---------------------------------------------------------------------------
# Lorenz
# ---------------------------------------------------------------------------

class TestLorenz:
    def _sys(self):
        return Lorenz()

    def test_dynamics_shape(self):
        out = self._sys().dynamics(0.0, np.array([1.0, 1.0, 1.0]))
        assert np.asarray(out).shape == (3,)

    def test_dynamics_finite(self):
        out = self._sys().dynamics(0.0, np.array([10.0, 10.0, 25.0]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 3

    def test_fixed_point_origin(self):
        """Origin (0,0,0) is always a fixed point for any parameters."""
        sys = Lorenz()
        dstate = sys.dynamics(0.0, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(dstate, 0.0, atol=1e-12)

    def test_sensitive_to_ic(self):
        """Two trajectories with different ICs diverge — hallmark of chaos.
        Uses a modest perturbation and standard tolerances so divergence is observable."""
        sys = Lorenz()
        ic1 = np.array([1.0, 0.0, 0.0])
        ic2 = ic1 + np.array([1e-3, 0.0, 0.0])
        # Use standard tolerances (not ultra-tight) so chaotic divergence is not suppressed
        sol1 = solve_ivp(sys.dynamics, (0.0, 25.0), ic1, method="RK45", rtol=1e-6, atol=1e-8)
        sol2 = solve_ivp(sys.dynamics, (0.0, 25.0), ic2, method="RK45", rtol=1e-6, atol=1e-8)
        final_dist = np.linalg.norm(sol1.y[:, -1] - sol2.y[:, -1])
        assert final_dist > 1.0, "Lorenz should show sensitive dependence on initial conditions"


# ---------------------------------------------------------------------------
# Van der Pol
# ---------------------------------------------------------------------------

class TestVanDerPol:
    def _sys(self):
        return VanDerPol()

    def test_dynamics_shape(self):
        out = self._sys().dynamics(0.0, np.array([1.0, 0.0]))
        assert np.asarray(out).shape == (2,)

    def test_dynamics_finite(self):
        out = self._sys().dynamics(0.0, np.array([2.0, 1.0]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 2

    def test_limit_cycle_amplitude(self):
        """Van der Pol oscillator converges to a limit cycle with amplitude ≈ 2."""
        sys = self._sys()
        ic = np.array([0.1, 0.0])
        sol = integrate(sys, ic, T=50.0)
        v = sol.y[0]
        amplitude = np.max(np.abs(v[-len(v)//4:]))  # last quarter of trajectory
        assert 1.5 < amplitude < 3.0, f"VdP limit cycle amplitude unexpected: {amplitude:.2f}"


# ---------------------------------------------------------------------------
# DC Motor
# ---------------------------------------------------------------------------

_MOTOR_PARAMS = dict(R=1.0, L=0.01, Ke=0.01, Kt=0.01, J=0.001, bm=0.0, V0=6.0)


class TestDCMotor:
    def _sys(self):
        return DCMotor(**_MOTOR_PARAMS)

    def test_dynamics_shape(self):
        out = self._sys().dynamics(0.0, np.array([0.0, 0.0]))
        assert np.asarray(out).shape == (2,)

    def test_dynamics_finite(self):
        out = self._sys().dynamics(0.0, np.array([1.0, 10.0]))
        assert np.all(np.isfinite(out))

    def test_state_labels(self):
        assert len(self._sys().state_labels()) == 2

    def test_step_response_steady_state(self):
        """Under constant voltage, motor reaches a non-zero steady-state speed."""
        sys = self._sys()
        ic = np.array([0.0, 0.0])
        sol = integrate(sys, ic, T=5.0)
        omega_final = sol.y[1, -1]
        assert omega_final > 0.1, "Motor should spin up under default voltage"
