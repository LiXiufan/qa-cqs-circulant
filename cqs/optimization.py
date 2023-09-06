########################################################################################################################
# Copyright (c) Xiufan Li. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Xiufan Li
# Supervisor: Patrick Rebentrost
# Institution: Centre for Quantum Technologies, National University of Singapore
# For feedback, please contact Xiufan at: e1117166@u.nus.edu.
########################################################################################################################


# !/usr/bin/env python3

"""
    Optimization module for solving the optimal combination parameters.
    In this module, we implement the CVXOPT package as an external resource package.
    CVXOPT is a free software package for convex optimization based on the Python programming language.
    Reference: https://cvxopt.org
    MIT Course: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    CVXOPT Notation of a quadratic optimization problem:
                min    1/2  x^T P x + q^T x
          subject to   Gx  <=  h
                       Ax  =  b
"""
from numpy import transpose, conj
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
from numpy import linalg, array, identity, diag, multiply, real, imag
from sympy import Matrix

def solve_combination_parameters(W, r, which_opt=None):

    if which_opt is None:
        which_opt = 'cvxopt'

    if which_opt == 'cvxopt':
        W = 2 * matrix(W)
        r = (-2) * matrix(r)
        # Solve
        comb_params = qp(W, r, kktsolver='ldl', options={'kktreg': 1e-12})['x']

    elif which_opt == 'inv':

        W = Matrix(W)
        P, D = W.diagonalize()
        D = array(D, dtype='complex128')
        D_diag = diag(D)
        D_diag_inv = []
        for d in D_diag:
            if linalg.norm(d) <= 1e-12:
                d_inv = 0
            else:
                if linalg.norm(real(d)) <= 1e-12:
                    d = 0 + imag(d) * 1j
                if linalg.norm(imag(d)) <= 1e-12:
                    d = real(d)
                d_inv = 1 / d
            D_diag_inv.append(d_inv)
        D_diag_inv = array(D_diag_inv, dtype='complex128')
        comb_params = multiply(D_diag_inv, r.reshape(-1))

    else:
        raise ValueError

    half_var = int(len(comb_params) / 2)
    vars = [0 for _ in range(half_var)]

    for i in range(half_var):
        var = comb_params[i] + comb_params[half_var + i] * 1j
        vars[i] = var

    params_array = array(comb_params).reshape(-1, 1)
    W_array = array(W / 2)
    r_array = array(r / (-2)).reshape(-1, 1)
    print(params_array)
    print(W_array)
    print(r_array)
    loss = abs((transpose(params_array) @ W_array @ params_array - 2 * transpose(r_array) @ params_array + 1).item())
    return loss, vars


