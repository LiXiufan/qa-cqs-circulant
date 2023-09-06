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
    This is the example of solving heat transfer problems corresponding to different condition number (by different ξ).
    The circulant matrix obtained by discretization of the heat transfer problem has the form of:

     - 2 - ξ         1              0        ....      0       1
        1         -2 - ξ            1        ....      0       0
        0            1           -2 - ξ      ....      0       0
       ...          ...            ...       ....     ...     ...
        0            0              0        ....   -2 - ξ     1
        1            0              0        ....      1    -2 - ξ

    Therefore, it is 1-banded and can be decomposed into linear combination of permutations:
                    C = (−2 − ξ)I + Q + Q^−1,
    where Q has the form of:

        0            0              0        ....      0       1
        1            0              0        ....      0       0
        0            1              0        ....      0       0
       ...          ...            ...       ....     ...     ...
        0            0              0        ....      0       0
        0            0              0        ....      1       0
"""

from cqs.generator import Circulant
from numpy import pi
from main import cqs_circulant_main

# 1. Problem Setting
# Generate the problem of solving a circulant linear systems of equations.
# Set the dimension of the circulant matrix C on the left hand side of the equation.
qubit_number = 5
dim = 2 ** qubit_number
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# Expected error range
error = 0.1
# Initialize the circulant matrix
C = Circulant(number_of_terms, dim, qubit_number)
print('Qubits are tagged as:', ['Q' + str(i) for i in range(C.get_width())])
# Set the '\xi' parameter
xi = 1
# Generate C in the following way
pows = [0, 1, dim-1]
coeffs = [- 2 - xi, 1, 1]
C.generate(permu_pows=pows, coeffs=coeffs)
print('Coefficients of the terms are:', coeffs)
print('Decomposed powers of permutations are:', pows)

# Unitary description U_b on the right hand side of the equation.
U_b = [[['H'], ['X'], ['H'], ['X'], ['Rx', pi / 5]]]
print('The circuit description of U_b is:', U_b)
# Simulation / hardware access
# access = 'ibmq-perth'
access = 'ibmq-statevector'
# Truncated threshold T
T = 2
loss, vars = cqs_circulant_main(C, U_b, T, access)
print(loss)
print(vars)





