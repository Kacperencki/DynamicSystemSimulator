import numpy as np

def AB_analytic(M, m, l, g=9.81):
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -(m * g) / M, 0],
                  [0, 0, 0, 1],
                  [0, 0, g * (M + m) / (M * l), 0]], dtype=float)
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]], dtype=float)
    return A, B

