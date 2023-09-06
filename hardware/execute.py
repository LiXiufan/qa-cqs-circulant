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
    This is the execution module to perform Hadamard tests.

    Here we can perform the Hadamard tests with various backends, ranging from classical simulation to quantum hardware.
"""
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.circuit.library import QFT
from numpy import pi
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider


def Hadamard_test_QFT(U_b, q_pow, alpha=1, access=None, shots=1024):
    if access is None:
        backend = 'qiskit-aer'

    elif access == 'qiskit-aer':
        backend = Aer.get_backend('statevector_simulator')

    elif access == 'ibmq-statevector':
        provider = IBMProvider(instance='ibm-q/open/main')
        hub = "ibm-q"
        group = "open"
        project = "main"
        backend_name = "simulator_statevector"
        backend = provider.get_backend(backend_name, instance=f"{hub}/{group}/{project}")

    elif access == 'ibmq-perth':
        provider = IBMProvider(instance='ibm-q/open/main')
        hub = "ibm-q"
        group = "open"
        project = "main"
        backend_name = "ibm_perth"
        backend = provider.get_backend(backend_name, instance=f"{hub}/{group}/{project}")

    else:
        raise ValueError

    width = len(U_b[0])
    ancilla = 1
    q_circuit = QuantumRegister(width, 'q')
    q_rot = QuantumRegister(width, 'q')
    q_had = QuantumRegister(width + ancilla, 'q')
    c_had = ClassicalRegister(1, 'c')

    U_b_cir = QuantumCircuit(q_circuit)
    for layer in U_b:
        for i, gate in enumerate(layer):
            name = gate[0]
            if name == 'X':
                U_b_cir.x(q_circuit[i])
            elif name == 'Y':
                U_b_cir.y(q_circuit[i])
            elif name == 'Z':
                U_b_cir.z(q_circuit[i])
            elif name == 'I':
                # circuit.i(i)
                U_b_cir.h(q_circuit[i])
                U_b_cir.h(q_circuit[i])
            elif name == 'H':
                U_b_cir.h(q_circuit[i])
            elif name == 'Rx':
                theta = gate[1]
                U_b_cir.rx(theta, q_circuit[i])
            elif name == 'Ry':
                theta = gate[1]
                U_b_cir.ry(theta, q_circuit[i])
            elif name == 'Rz':
                theta = gate[1]
                U_b_cir.rz(theta, q_circuit[i])

    U_b_gate = U_b_cir.to_gate()
    qft_gate = QFT(num_qubits=width, inverse=False, name='qft').to_gate()
    rot_cir = QuantumCircuit(q_rot)
    Theta = [(2 * (2 ** i) * q_pow * pi) / (2 ** width) for i in range(width)]
    for i in range(width):
        rot_cir.p(Theta[i], q_rot[i])
    rot_cir_gate = rot_cir.to_gate()
    C_rot_cir_gate = rot_cir_gate.control()

    Hadamard_circuit = QuantumCircuit(q_had, c_had)
    Hadamard_circuit.h(q_had[0])
    if alpha == 1j:
        Hadamard_circuit.s(q_had[0])
    Hadamard_circuit.append(U_b_gate, [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.append(qft_gate, [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.append(C_rot_cir_gate, [q_had[0]] + [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.h(q_had[0])
    Hadamard_circuit.measure([q_had[0]], [c_had[0]])
    # Transpile the circuit for Hadamard test
    result = transpile(Hadamard_circuit, backend)
    job = backend.run(result, shots=shots)
    return backend, job.job_id()
