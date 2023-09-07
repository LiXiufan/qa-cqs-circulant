"""
    This is the main function of the CQS approach for circulant matrix.
"""

from circulant_solver.calculation import calculate_W_r
from circulant_solver.optimization import solve_combination_parameters

def cqs_circulant_main(C, U_b, T, access, shots=1024):
    # Obtain the Ansatz basis
    Ansatz_pows = list(range(-T, T + 1))
    W, r = calculate_W_r(C, U_b, Ansatz_pows, access=access, shots=shots)
    loss, results = solve_combination_parameters(W, r)
    return loss, results
