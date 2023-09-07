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
from circulant_solver.algo import cqs_circulant_main
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# 1. Problem Setting
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# shot budget per Hadamard test
shots = 10 ** 5
# Set the '\xi' parameter
xi = 0.01
# Initialize the circulant matrix
pows = [0, 1, -1]
coeffs = [- 2 - xi, 1, 1]
C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
print('Coefficients of the terms are:', coeffs)
print('Decomposed powers of permutations are:', pows)

# Unitary description U_b on the right hand side of the equation.
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
circuit = QuantumCircuit(qreg_q)

circuit.h(qreg_q[1])
circuit.h(qreg_q[3])
circuit.h(qreg_q[0])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.h(qreg_q[0])
circuit.cx(qreg_q[2], qreg_q[3])

U_b = circuit

print('The circuit description of U_b is:', U_b)
# Simulation / hardware access
access = "true"
# Truncated threshold T
T = 15
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
