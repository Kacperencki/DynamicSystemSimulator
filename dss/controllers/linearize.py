from __future__ import annotations

import numpy as np


def linearize_upright(system, include_damping: bool = True, include_pivot_input: bool = True):
    """
    Linearize the cart–pole around the upright equilibrium (theta = 0).

    State:
        x = [x, x_dot, theta, theta_dot]^T
    Inputs:
        - F_cart  : horizontal force on the cart
        - T_pivot : external pivot torque added to tau (optional)

    This linearization matches InvertedPendulum._core() conventions:
        tau = m*g*lc*sin(theta) + pend_fric + pend_drive + T_ext
    where pend_fric includes viscous damping as (-b_pend * theta_dot).
    """
    M = float(system.M)
    m = float(system.m)
    g = float(system.g)
    lc = float(system.lc)  # pivot → COM
    Ip = float(system.Ip)  # inertia about pivot

    b_cart = float(getattr(system, "b_cart", 0.0))
    b_pend = float(getattr(system, "b_pend", 0.0))

    # Denominator at θ = 0 (cosθ ≈ 1, sinθ ≈ 0)
    den0 = (M + m) * Ip - (m * lc) ** 2
    if abs(den0) < 1e-12:
        den0 = 1e-12 if den0 >= 0 else -1e-12

    # Gravity coupling (upright is unstable)
    a23 = -(m**2 * g * lc**2) / den0
    a43 = (m * (M + m) * g * lc) / den0

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, a23, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, a43, 0.0],
        ],
        dtype=float,
    )

    if include_damping:
        # Cart viscous damping enters through F_eff = ... - b_cart * x_dot
        if b_cart != 0.0:
            A[1, 1] += -(Ip * b_cart) / den0
            A[3, 1] += (m * lc * b_cart) / den0

        # Pend viscous damping enters through tau = ... - b_pend * theta_dot
        if b_pend != 0.0:
            A[1, 3] += (m * lc * b_pend) / den0
            A[3, 3] += -((M + m) * b_pend) / den0

    # Input matrices
    b2_F = Ip / den0
    b4_F = -(m * lc) / den0

    if include_pivot_input:
        # T_pivot is added to tau with positive sign
        b2_T = -(m * lc) / den0
        b4_T = (M + m) / den0
        B = np.array(
            [
                [0.0, 0.0],
                [b2_F, b2_T],
                [0.0, 0.0],
                [b4_F, b4_T],
            ],
            dtype=float,
        )
    else:
        B = np.array(
            [
                [0.0],
                [b2_F],
                [0.0],
                [b4_F],
            ],
            dtype=float,
        )

    return A, B
