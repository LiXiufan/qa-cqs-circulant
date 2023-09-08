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
from main import cqs_circulant_main
import matplotlib.pyplot as plt
import numpy as np
import sys
from qiskit import QuantumRegister, QuantumCircuit

np.set_printoptions(threshold=sys.maxsize)

# 1. Problem Setting
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# shot budget per Hadamard test
shots = 10 ** 5
# Set the '\xi' parameter
xi = 0.2
# Initialize the circulant matrix
pows = [0, 1, -1]
coeffs = [- 2 - xi, 1, 1]
C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
print('Coefficients of the terms are:', coeffs)
print('Decomposed powers of permutations are:', pows)

# QAOA Circuit
n = 5
d = 1
theta = [np.pi / (2 ** i) for i in range(1, n + 1)]
# theta = [np.random.rand() * 2 * np.pi for _ in range(d * n)]
print(theta)
qreg_q = QuantumRegister(n, 'q')
circuit = QuantumCircuit(qreg_q)
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

# Simulation / hardware access
# access = "true"
access = "qiskit-aer"
# access = 'ibmq-statevector'
# access = 'ibmq-perth'

# Truncated threshold T
T = 6
# Record file
file_name = 'cqs_circulant_example_1'
loss_list = []
results_list = []
T_List = []
for t in range(1, T + 1):
    T_List.append(t)
    loss, results = cqs_circulant_main(C, U_b, t, access=access, shots=shots)
    loss_list.append(loss)
    results_list.append(results)

plt.title("CQS: Loss - Depth", fontsize=10)
plt.plot(T_List, loss_list, 'g-', linewidth=2.5, label='Loss Function - Iteration')
lgd = plt.legend() # NB different 'prop' argument for legend
# lgd = plt.legend(fontsize=20) # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(T_List, T_List)
plt.xlabel("Truncated Threshold", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.show()
