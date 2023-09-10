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
from time import strftime, localtime
from functools import partial
from multiprocessing import Pool

np.set_printoptions(threshold=sys.maxsize)

# 1. Problem Setting
# Set the number of permutations in the circulant matrix C
number_of_terms = 3
# shot budget per Hadamard test
shots = 1e+6
# Set the '\xi' parameter
xi = 0.2
# Initialize the circulant matrix
pows = [0, 1, -1]
coeffs = [- 2 - xi, 1, 1]
C = Circulant(number_of_terms, permu_pows=pows, coeffs=coeffs)
print('Coefficients of the terms are:', coeffs)
print('Decomposed powers of permutations are:', pows)

qreg_q = QuantumRegister(3, 'q')
circuit = QuantumCircuit(qreg_q)
U_b = circuit

print('The circuit description of U_b is:')
print(U_b)

# Simulation / hardware access
# access = "true"
access = "qiskit-aer"
# access = 'ibmq-statevector'
# access = 'ibmq-perth'
# access = "sample"

# Truncated threshold T
T = 5
# Record file
log_file = f"heat_transfer_{strftime('%Y%m%d%H%M%S', localtime())}"
T_List = list(range(1, T + 1))
output = cqs_circulant_main(C, U_b, T_List, access, shots, log_file)
loss_list = [item[0] for item in output]
results_list = [item[1] for item in output]

plt.title("CQS: Loss - Depth", fontsize=10)
plt.plot(T_List, loss_list, 'g-', linewidth=2.5, label='Loss Function - Iteration')
lgd = plt.legend()  # NB different 'prop' argument for legend
# lgd = plt.legend(fontsize=20) # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(T_List, T_List)
plt.xlabel("Truncated Threshold", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.savefig(f"{log_file}.png")
