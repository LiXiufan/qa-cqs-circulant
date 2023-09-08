"""
    This is the main function of the CQS approach for circulant matrix.
"""

__all__ = [
    "cqs_circulant_main"
]

from circulant_solver.logger import log
from circulant_solver.circulant import Circulant
from circulant_solver.inner_product import InnerProduct
from circulant_solver.calculation import calculate_W_r
from circulant_solver.optimization import solve_combination_parameters
import numpy as np
from typing import Union, List

def cqs_circulant_main(C:Circulant, U_b, T: Union[int, List[int]], access, shots=1024, logfile=None):
    # Obtain the Ansatz basis
    if isinstance(T, list):
        max_T = np.max(T)
    else:
        max_T = T
        T = [max_T]
    K = np.max(np.abs(C.get_pows()))
    ip = InnerProduct(access, U_b, K, max_T, shots)
    results = []
    for t in T:
        W, r = calculate_W_r(C, list(range(-t, t + 1)), ip)
        loss, alpha = solve_combination_parameters(W, r)
        if logfile is not None:
            log(C, U_b, W, r, t, alpha, loss, access, shots, logfile)
        results.append((loss, alpha))
    return results