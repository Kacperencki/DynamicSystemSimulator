import numpy as np

def ideal_AB(M, m, l, g=9.81):
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -(m * g) / M, 0],
                  [0, 0, 0, 1],
                  [0, 0, g * (M + m) / (M * l), 0]], dtype=float)
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]], dtype=float)
    return A, B


def damped_AB(M, m, l, g=9.81, b_cart=0.0, b_pend=0.0):
    A, B = ideal_AB(M, m, l, g)

    A[1,1] = -b_cart / M # somethin wrong here
    A[3,1] = -b_pend / (m * l**2)

    return A, B