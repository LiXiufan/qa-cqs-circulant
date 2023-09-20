# !/usr/bin/env python3
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from typing import List, Tuple

__all__ = [
    "solve_combination_parameters"
]


def solve_combination_parameters(W: np.ndarray, r: np.ndarray) -> Tuple[float, List]:
    r"""Optimization module for solving the optimal combination parameters.

    In this module, we implement the CVXOPT package as an external resource package.
    CVXOPT is a free software package for convex optimization based on the Python programming language.
    Reference: https://cvxopt.org
    MIT Course: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    CVXOPT Notation of a quadratic optimization problem:
                min    1/2  x^T P x + q^T x
          subject to   Gx  <=  h
                       Ax  =  b

    Args:
        W (np.ndarray): the auxiliary matrix W
        r (np.ndarray): the auxiliary vector r

    Returns:
        Tuple[float, List]: loss and the optimal combination parameters
    """
    W = 2 * matrix(W)
    r = (-2) * matrix(r)
    # Solve the optimization problem using the kkt solver with regularization constant of 1e-12
    # Note: for more realistic experiments, due to the erroneous results,
    # it is suggested to change the regularization constant to get a better performance.
    comb_params = qp(W, r, kktsolver='ldl', options={'kktreg': 1e-12})['x']

    half_var = int(len(comb_params) / 2)
    results = [0 for _ in range(half_var)]

    for i in range(half_var):
        var = comb_params[i] + comb_params[half_var + i] * 1j
        results[i] = var

    params_array = np.array(comb_params).reshape(-1, 1)
    W_array = np.array(W / 2)
    r_array = np.array(r / (-2)).reshape(-1, 1)
    loss = abs(
        (np.transpose(params_array) @ W_array @ params_array - 2 * np.transpose(r_array) @ params_array + 1).item())
    return loss, results
