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
# access = "qiskit-aer"
# access = 'ibmq-statevector'
# access = 'ibmq-perth'
# access = "sample"

N = 8
# print(theta)

# for i in range(n):
#     circuit.x(qreg_q[i])
# for i in range(n):
#     circuit.h(qreg_q[i])
# for j in range(d):
#     for i in range(n - 1):
#         circuit.cx(qreg_q[i], qreg_q[i + 1])
#         circuit.rz(theta[i], qreg_q[i + 1])
#         circuit.cx(qreg_q[i], qreg_q[i + 1])
#     circuit.cx(qreg_q[n - 1], qreg_q[0])
#     circuit.rz(theta[n - 1], qreg_q[0])
#     circuit.cx(qreg_q[n - 1], qreg_q[0])
# print('The circuit description of U_b is:')
# print(U_b)

# Set the '\xi' parameter
# XI = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
# Record file
log_file = f"heat_transfer_{strftime('%Y%m%d%H%M%S', localtime())}"
# XI = [0.01, 0.02, 0.04, 0.1, 0.12, 0.2, 0.5, 1.0, 2.0, 4.0]
XI = [0.2]
T_list = []
# Cond_list = []
for xi in XI:
    N_list = []
    for n in range(2, N + 1):
        N_list.append(n)
        # System size
        # QAOA Circuit
        d = 1
        theta = [np.pi / (2 ** i) for i in range(1, n + 1)]
        qreg_q = QuantumRegister(n, 'q')
        circuit = QuantumCircuit(qreg_q)
        for i in range(n):
            circuit.x(qreg_q[i])
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

        # cond_num = (xi + 4) / xi
        # Cond_list.append(cond_num)
        # Initialize the circulant matrix
        pows = [0, 1, -1]
        coeffs = [- 2 - xi, 1, 1]
        C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
        print('Coefficients of the terms are:', coeffs)
        print('Decomposed powers of permutations are:', pows)
        t = cqs_circulant_cond_main(C, U_b, access, shots, log_file)
        T_list.append(t)

print(T_list)
plt.title("Figure: Truncated Threshold - System Size", fontsize=10)
plt.plot(N_list, T_list, 'g-', linewidth=2.5, label="Truncated Threshold - System Size")
lgd = plt.legend()  # NB different 'prop' argument for legend
# lgd = plt.legend(fontsize=20) # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(N_list, N_list)
plt.xlabel("System Size '\Xi'", fontsize=10)
plt.ylabel("Truncated Threshold", fontsize=10)
plt.savefig(f"{log_file}.png")

