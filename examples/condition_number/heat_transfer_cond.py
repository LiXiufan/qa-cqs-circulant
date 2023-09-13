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

    In this experiment, we study the relationship between the condition number and the truncation threshold.
    The vector b is set to be tensor product of zero states.
    Experiments are conducted using matrix multiplication to get rid of the shot noise.
    Impressive discovery is that for this specific task, the truncation threshold is upper bounded by a constant
    depth of 16 even in the case of large condition numbers.
    We conjecture that this may be due to the fact that the effective condition number is small in this problem.
"""
from circulant_solver.circulant import Circulant
from main import cqs_circulant_cond_main
import matplotlib.pyplot as plt
import numpy as np
import sys
from qiskit import QuantumRegister, QuantumCircuit
from time import strftime, localtime

np.set_printoptions(threshold=sys.maxsize)

# 1. Problem Setting
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# shot budget per Hadamard test
shots = 10 ** 6
# Simulation / hardware access
access = "true"
# System size
N = 5
qreg_q = QuantumRegister(N, 'q')
circuit = QuantumCircuit(qreg_q)
U_b = circuit
print('The circuit description of U_b is:')
print(U_b)

# We conduct multiple experiments by selecting different '\xi' parameters
# XI = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
# XI = [0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01,
#       0.04, 0.07, 0.1, 0.12, 0.15, 0.2, 0.5, 1.0, 2.0]
# XI = [0.2, 0.5, 1.0]
XI = [0.0125, 0.013, 0.014, 0.015]

# Record file
log_file = f"heat_transfer_{strftime('%Y%m%d%H%M%S', localtime())}"

# Initialize
T_list = []
Cond_list = []

# Use the algorithm to solve the circulant linear systems with different condition numbers
for xi in XI:
    cond_num = (xi + 4) / xi
    print("Condition number is:", cond_num)
    # Initialize the circulant matrix
    pows = [0, 1, -1]
    coeffs = [- 2 - xi, 1, 1]
    C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
    print("Coefficients of the terms are:", coeffs)
    print("Decomposed powers of permutations are:", pows)
    t = cqs_circulant_cond_main(C, U_b, access, shots, log_file)
    Cond_list.append(cond_num)
    T_list.append(t)
print("Truncation thresholds are:", T_list)

# Plot and record the results
plt.title("Figure: Truncation threshold - condition number", fontsize=10)
plt.plot(XI, T_list, 'g-', linewidth=2.5, label="Truncation threshold - condition number")
lgd = plt.legend()  # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(XI, XI)
plt.xlabel("Condition number", fontsize=10)
plt.ylabel("Truncation threshold", fontsize=10)
plt.savefig(f"{log_file}.png")
