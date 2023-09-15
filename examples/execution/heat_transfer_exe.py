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

    We test the performance of our algorithm in solving this circulant linear systems with different backends
    in simulations and 'ibm-perth' quantum hardware.
"""
from circulant_solver.circulant import Circulant
from main import cqs_circulant_main
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
# Set the '\xi' parameter
xi = 0.2
# Initialize the circulant matrix
pows = [0, 1, -1]
coeffs = [- 2 - xi, 1, 1]
C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
print("Coefficients of the terms are:", coeffs)
print("Decomposed powers of permutations are:", pows)

# Circuit for preparation of b
n = 5
qreg_q = QuantumRegister(n, 'q')
circuit = QuantumCircuit(qreg_q)
# Choose the circuit type
cir_b = 'QAOA'
# cir_b = 'identity'

if cir_b == 'QAOA':
    d = 1
    theta = [np.pi / (2 ** i) for i in range(1, n + 1)]
    print("The rotation parameters for QAOA circuit are:", theta)
    # QAOA embedding
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
    print('The circuit description of U_b is:')
    print(U_b)
elif cir_b == 'identity':
    U_b = circuit
    print('The circuit description of U_b is:')
    print(U_b)
else:
    raise ValueError

# Simulation / hardware access
# access = "true"
access = "qiskit-aer"
# access = 'ibmq-statevector'
# access = 'ibmq-perth'
# access = "sample"

# Truncated threshold T
T = 6
# Record file
# log_file = f"heat_transfer_{strftime('%Y%m%d%H%M%S', localtime())}"
# Close the logging file
log_file = None

# Use the algorithm to solve the circulant linear systems
T_List = list(range(1, T + 1))
print("Truncation thresholds are:", T_List)
output = cqs_circulant_main(C, U_b, T_List, access, shots, logfile=log_file)

# Get the loss and results
loss_list = [item[0] for item in output]
print("Loss for each truncation threshold:", loss_list)
results_list = [item[1] for item in output]
print("The results are:", loss_list)

# Plot and record the results
plt.title("Figure: Loss - truncation threshold", fontsize=10)
plt.plot(T_List, loss_list, 'g-', linewidth=2.5, label='Loss - truncation threshold')
lgd = plt.legend()  # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(T_List, T_List)
plt.xlabel("Truncation threshold", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.show()
# plt.savefig(f"{log_file}.png")
