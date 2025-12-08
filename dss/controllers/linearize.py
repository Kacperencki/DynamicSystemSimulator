import numpy as np

def linearize_upright(system, include_damping=True, include_pivot_input=True):
    M  = float(system.M)
    m  = float(system.m)
    g  = float(system.g)
    lc = float(system.lc)              # pivot → COM
    Ip = float(system.Ip)              # inertia about pivot
    b_cart = float(getattr(system, "b_cart", 0.0))
    b_pend = float(getattr(system, "b_pend", 0.0))

    # Denominator at θ = 0 (cosθ ≈ 1, sinθ ≈ 0)
    den0 = (M + m) * Ip - (m * lc) ** 2
    if abs(den0) < 1e-12:
        den0 = 1e-12 if den0 >= 0 else -1e-12

    # ----- A matrix (small-angle, no inputs) -----
    # From your exact model:
    # ẍ = [ Ip*F  - (m lc) τ ] / den
    # θ̈ = [-(m lc)F + (M+m) τ] / den, with τ = m g lc θ (linearized)
    # So the θ-coefficients are:
    a23 = -(m**2 * g * lc**2) / den0
    a43 =  (m * (M + m) * g * lc) / den0

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a23, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, a43, 0.0],
    ], dtype=float)

    if include_damping:
        # Linear viscous terms near zero (Coulomb omitted)
        A[1, 1] += -(Ip * b_cart) / den0
        A[3, 3] += -((M + m) * b_pend) / den0

    # ----- B matrix: columns = [F_cart, T_pivot] -----
    b2_F =  Ip / den0
    b4_F = -(m * lc) / den0

    if include_pivot_input:
        b2_T =  (m * lc) / den0
        b4_T = -(M + m) / den0
        B = np.array([[0.0, 0.0],
                      [b2_F, b2_T],
                      [0.0, 0.0],
                      [b4_F, b4_T]], dtype=float)
    else:
        B = np.array([[0.0],
                      [b2_F],
                      [0.0],
                      [b4_F]], dtype=float)

    return A, B
