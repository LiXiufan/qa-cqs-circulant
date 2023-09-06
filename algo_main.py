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
    This is the main function of the CQS approach for circulant matrix.
"""

from cqs.calculation import calculate_W_r
from cqs.optimization import solve_combination_parameters

def cqs_circulant_main(C, U_b, T, access=None, shots=1024, file_name='message.txt'):

    # Obtain the Ansatz basis
    Ansatz_pows = list(range(-1 * T, T + 1))
    print("\n")
    print("Truncated threshold:", T, " Powers of Ansatz permutations are:\n", Ansatz_pows)
    W, r = calculate_W_r(C, U_b, Ansatz_pows, access=access, shots=shots, file_name=file_name)
    # Record W and r
    file1 = open(file_name, "a")
    file1.writelines(
        ['\n', "Truncated threshold:", str(T), " Matrix W is:\n", str(W), "\nTruncated threshold:", str(T), " Vector r is:\n",
         str(r),
         '\n\n'])
    file1.close()
    # Solve the optimization of combination parameters: x* = \sum (alpha * ansatz_state)
    loss, vars = solve_combination_parameters(W, r, which_opt=None)
    file1 = open(file_name, "a")
    file1.writelines(["\nTruncated threshold:", str(T), " Combination parameters are:", str(vars), '\n\n'])
    file1.writelines(['\nTruncated threshold:', str(T), " Loss:", str(loss), '\n\n'])
    file1.close()
    print("Truncated threshold:", T, " Combination parameters are:\n", vars)
    print("Truncated threshold:", T, " Loss:\n", loss)
    return loss, vars

















