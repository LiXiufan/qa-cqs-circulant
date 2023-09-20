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
    Experiments are conducted using the sparse matrix estimator to get rid of the shot noise.
"""
from circulant_solver.circulant import Circulant
from main import cqs_circulant_cond_main
import matplotlib.pyplot as plt
import numpy as np
import sys
from qiskit import QuantumRegister, QuantumCircuit

np.set_printoptions(threshold=sys.maxsize)

# 1. Problem Setting
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# shot budget per Hadamard test
shots = 10 ** 6
# Simulation / hardware access
access = "true"
# System size
# Circuit for preparation of b
n = 15
qreg_q = QuantumRegister(n, 'q')
circuit = QuantumCircuit(qreg_q)
# Choose the circuit type
# cir_b = 'QAOA'
cir_b = 'identity'
# cir_b = 'vector'
if cir_b == 'QAOA':
    d = 1
    theta = [np.pi / (2 ** i) for i in range(1, n + 1)]
    print("The rotation parameters for QAOA circuit are:", theta)
    # QAOA embedding
    for i in range(n):
        circuit.h(qreg_q[i])
    for j in range(d):
        for i in range(n - 1):
            circuit.cx(qreg_q[i], qreg_q[i + 1])
            circuit.rz(theta[i], qreg_q[i + 1])
            circuit.cx(qreg_q[i], qreg_q[i + 1])
        circuit.cx(qreg_q[n - 1], qreg_q[0])
        circuit.rz(theta[n - 1], qreg_q[0])
        circuit.cx(qreg_q[n - 1], qreg_q[0])
    U_b = circuit
    print('The circuit description of U_b is:')
    print(U_b)
elif cir_b == 'identity':
    U_b = circuit
    print('The circuit description of U_b is:')
    print(U_b)
elif cir_b == 'vector':
    U_b = np.zeros(2 ** n)
else:
    raise ValueError

# We conduct multiple experiments by selecting different '\xi' parameters
# XI = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
XI = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
# XI = [0.2, 0.5, 1.0]
# XI = [0.0125, 0.013, 0.014, 0.015]

# Record file
# log_file = f"heat_transfer_{strftime('%Y%m%d%H%M%S', localtime())}"
# Close the log file
log_file = None

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
print("Condition numbers are:", Cond_list)
print("Truncation thresholds are:", T_list)

# Plot and record the results
plt.title("Figure: Truncation threshold - condition number", fontsize=10)
plt.plot(Cond_list, T_list, 'g-', linewidth=2.5, label="Truncation threshold - condition number")
lgd = plt.legend()  # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(Cond_list, Cond_list)
plt.xlabel("Condition number", fontsize=10)
plt.ylabel("Truncation threshold", fontsize=10)
plt.show()
# plt.savefig(f"{log_file}.png")
