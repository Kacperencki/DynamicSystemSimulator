# dss/models/inverted_pendulum.py
"""
Cart-pole (inverted pendulum on a cart) dynamical model.

Equations of motion (Lagrangian derivation, linearised inertia matrix)
-----------------------------------------------------------------------
State:  [x, ẋ, θ, θ̇]
Input:  F  (horizontal cart force, positive = rightward)

The coupled ODEs come from the Lagrangian of the cart-pole system:

  |M+m    m·lc·cosθ| [ẍ  ]   |F + friction_cart + drive_cart − m·lc·sinθ·θ̇²|
  |m·lc·cosθ    Ip | [θ̈ ] = |m·g·lc·sinθ + friction_pend + drive_pend       |

Solved by Cramer's rule (see _core()):
  ẍ  = (Ip·F_eff − m·lc·cosθ · τ) / det
  θ̈  = (−m·lc·cosθ · F_eff + (M+m) · τ) / det
  det = (M+m)·Ip − (m·lc·cosθ)²

Sign convention:  θ=0 is upright, θ=π is hanging down.
                  Positive θ tilts the pole to the right.

Coulomb friction smoothing
--------------------------
sign(v) is replaced by tanh(coulomb_k·v) to avoid discontinuities that
can stall adaptive ODE solvers.  Higher coulomb_k → sharper approximation.
Default 1e3 is sharp enough for typical parameter ranges.

To add a new mode
-----------------
1. Add a tuple (use_cart_damp, use_pend_damp, use_cart_drive, use_pend_drive)
   to _MODE_FLAGS inside dynamics().
2. Document the mode string in the class docstring.
"""

import numpy as np


class InvertedPendulum:
    """
    Cart–pole (inverted pendulum on a cart).

    State (absolute):
        x, x_dot, theta, theta_dot

        x        : cart position [m] (right positive)
        x_dot    : cart velocity [m/s]
        theta    : pole angle from the UPWARD vertical [rad]
                   (theta = 0 is upright)
        theta_dot: pole angular velocity [rad/s]

    Mass model (geometry + inertia):
        - 'point'   : point mass at tip
                      lc = l, I_com = 0
        - 'uniform' : slender uniform rod
                      lc = l/2, I_com = (1/12) * m * l^2
        In the equations we use inertia about the pivot:
            Ip = I_com + m * lc^2

    Modes (which terms are active):
        - 'ideal'        : no friction, no drives
        - 'damped_cart'  : cart friction only
        - 'damped_pend'  : pendulum friction only
        - 'damped_both'  : cart + pendulum friction
        - 'driven'       : friction + harmonic drives + external inputs
        - 'dc_driven'    : friction + external inputs (no harmonic drives)

    External inputs (used in 'driven' / 'dc_driven'):
        dynamics(t, state, inputs)
            inputs is one of:
              - None             -> F_ext = 0, T_ext = 0
              - scalar           -> F_ext = inputs, T_ext = 0
              - (F_ext, T_ext)   -> both explicitly given
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        mode="damped_both",
        length=0.3,
        mass=0.2,
        cart_mass=0.5,
        gravity=9.81,
        # mass model
        mass_model="point",    # "point" or "uniform"
        I_com=None,            # inertia about COM (optional override)
        lc=None,               # pivot → COM (optional override)
        # friction (viscous + Coulomb)
        b_cart=0.0,
        coulomb_cart=0.0,
        b_pend=0.0,
        coulomb_pend=0.0,
        # harmonic drives (cart force, pivot torque)
        cart_drive_amp=0.0,
        cart_drive_freq=0.0,
        cart_drive_phase=0.0,
        pend_drive_amp=0.0,
        pend_drive_freq=0.0,
        pend_drive_phase=0.0,
        # Coulomb smoothing gain
        coulomb_k=1e3,
    ):
        # basic parameters
        self.l = float(length)
        self.m = float(mass)
        self.M = float(cart_mass)
        self.g = float(gravity)

        if self.l <= 0 or self.m <= 0 or self.M <= 0:
            raise ValueError("length, mass, cart_mass must be > 0")

        # mode
        self.mode = str(mode).lower().strip()
        valid_modes = {
            "ideal",
            "damped_cart",
            "damped_pend",
            "damped_both",
            "driven",
            "dc_driven",
        }
        if self.mode not in valid_modes:
            raise ValueError(
                "mode must be one of {}, got '{}'".format(
                    sorted(valid_modes), self.mode
                )
            )

        # mass model / geometry
        self.mass_model = str(mass_model).lower().strip()
        default_lc, default_Ic = self._mass_model_defaults(
            self.mass_model, self.m, self.l
        )

        # if user passes lc or I_com, they override defaults
        self.lc = default_lc if lc is None else float(lc)
        self.Ic = default_Ic if I_com is None else float(I_com)
        # inertia about pivot
        self.Ip = self.Ic + self.m * (self.lc ** 2)

        # friction parameters
        self.b_cart = float(b_cart)
        self.Fc_cart = float(coulomb_cart)
        self.b_pend = float(b_pend)
        self.Tc_pend = float(coulomb_pend)
        self.coulomb_k = float(coulomb_k)

        # harmonic drive parameters (cart force)
        self.A_cart = float(cart_drive_amp)
        self.f_cart = float(cart_drive_freq)
        self.phi_cart = float(cart_drive_phase)

        # harmonic drive parameters (pivot torque)
        self.A_pend = float(pend_drive_amp)
        self.f_pend = float(pend_drive_freq)
        self.phi_pend = float(pend_drive_phase)

    # ------------------------------------------------------------------ #
    # Public API (same style as other models)
    # ------------------------------------------------------------------ #
    def dynamics(self, t, state, inputs=None):
        """
        Unified entry point.

        inputs:
          - None           -> F_ext=T_ext=0
          - scalar         -> F_ext=inputs, T_ext=0
          - (F_ext, T_ext) -> both explicitly set
        """
        # decode external inputs
        if inputs is None:
            F_ext, T_ext = 0.0, 0.0
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            F_ext, T_ext = float(inputs[0]), float(inputs[1])
        else:
            F_ext, T_ext = float(inputs), 0.0

        # (use_cart_damp, use_pend_damp, use_cart_drive, use_pend_drive)
        _MODE_FLAGS = {
            "ideal":       (False, False, False, False),
            "damped_cart": (True,  False, False, False),
            "damped_pend": (False, True,  False, False),
            "damped_both": (True,  True,  False, False),
            "driven":      (True,  True,  True,  True),
            "dc_driven":   (True,  True,  False, False),
        }

        flags = _MODE_FLAGS.get(self.mode)
        if flags is None:
            raise RuntimeError("Unhandled mode: {!r}".format(self.mode))

        return self._core(
            t, state,
            use_cart_damp=flags[0],
            use_pend_damp=flags[1],
            use_cart_drive=flags[2],
            use_pend_drive=flags[3],
            F_ext=F_ext,
            T_ext=T_ext,
        )

    def state_labels(self):
        return ["x [m]", "x_dot [m/s]", "theta [rad]", "theta_dot [rad/s]"]

    def positions(self, state):
        """
        Positions for drawing (pivot on cart top, and pole tip).
        Theta = 0 is upright (tip above pivot).
        """
        x, _, theta, _ = state
        pivot = (float(x), 0.0)
        tip_x = x + self.l * np.sin(theta)
        tip_y = self.l * np.cos(theta)
        tip = (float(tip_x), float(tip_y))
        return [pivot, tip]

    def energy_check(self, state):
        """
        Returns [T, V, E].

        Potential energy is zero at upright (theta = 0).
        """
        x, x_dot, theta, theta_dot = state

        # COM linear velocity (pivot is on the cart)
        vx_com = x_dot + self.lc * theta_dot * np.cos(theta)
        vy_com = -self.lc * theta_dot * np.sin(theta)

        T_cart = 0.5 * self.M * (x_dot ** 2)
        T_pole = 0.5 * self.m * (vx_com ** 2 + vy_com ** 2) + 0.5 * self.Ic * (theta_dot ** 2)
        T_tot = T_cart + T_pole

        # Upright (theta=0) is zero potential → V = m*g*(lc*cosθ - lc)
        V = self.m * self.g * (self.lc * np.cos(theta) - self.lc)

        return np.array([float(T_tot), float(V), float(T_tot + V)], dtype=float)

    # optional: alias used in some code
    def energy(self, state):
        return self.energy_check(state)

    def joint_speed(self, state):
        """
        Joint speed for coupling with other models (e.g. DC motor).
        Here: pole angular speed theta_dot.
        """
        return float(state[3])

    # ------------------------------------------------------------------ #
    # Core dynamics (all physics terms in one place)
    # ------------------------------------------------------------------ #
    def _core(
        self,
        t,
        state,
        use_cart_damp,
        use_pend_damp,
        use_cart_drive,
        use_pend_drive,
        F_ext,
        T_ext,
    ):
        x, x_dot, theta, theta_dot = state
        m, M, g, lc, Ip = self.m, self.M, self.g, self.lc, self.Ip

        # Smooth sign for Coulomb (avoids chatter near zero)
        def sign_smooth(v):
            return float(np.tanh(self.coulomb_k * v))

        # friction forces / torques
        if use_cart_damp:
            cart_fric = -self.b_cart * x_dot - self.Fc_cart * sign_smooth(x_dot)
        else:
            cart_fric = 0.0

        if use_pend_damp:
            pend_fric = -self.b_pend * theta_dot - self.Tc_pend * sign_smooth(theta_dot)
        else:
            pend_fric = 0.0

        # harmonic drives (optional)
        if use_cart_drive and (self.A_cart != 0.0) and (self.f_cart != 0.0):
            cart_drive = self.A_cart * np.cos(self.f_cart * t + self.phi_cart)
        else:
            cart_drive = 0.0

        if use_pend_drive and (self.A_pend != 0.0) and (self.f_pend != 0.0):
            pend_drive = self.A_pend * np.cos(self.f_pend * t + self.phi_pend)
        else:
            pend_drive = 0.0

        # effective RHS terms
        cart_centripetal = m * lc * np.sin(theta) * (theta_dot ** 2)
        F_eff = F_ext + cart_fric + cart_drive + cart_centripetal

        # tau: torque about the pivot
        # pend_fric is already "opposes motion" (negative for +theta_dot), so ADD it.
        # External / drive torques should also ADD (positive torque increases theta).
        tau = m*g*lc*np.sin(theta) + pend_fric + pend_drive + T_ext


        # denominator (determinant of inertia matrix)
        den = (M + m) * Ip - (m * lc * np.cos(theta)) ** 2
        if abs(den) < 1e-9:
            den = 1e-9 if den >= 0 else -1e-9

        x_ddot = (Ip * F_eff - (m * lc * np.cos(theta)) * tau) / den
        theta_ddot = (-(m * lc * np.cos(theta)) * F_eff + (M + m) * tau) / den

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

    # ------------------------------------------------------------------ #
    # Helper: mass model defaults
    # ------------------------------------------------------------------ #
    def _mass_model_defaults(self, model, m, l):
        """
        Return (lc, I_com) for the given mass model.
        model: "point" or "uniform".
        """
        model = str(model).lower().strip()
        if model == "point":
            lc = l
            I_com = 0.0
        elif model == "uniform":
            lc = 0.5 * l
            I_com = (1.0 / 12.0) * m * (l ** 2)
        else:
            raise ValueError(
                "Unknown mass_model '{}'. Use 'point' or 'uniform'.".format(model)
            )
        return lc, I_com
