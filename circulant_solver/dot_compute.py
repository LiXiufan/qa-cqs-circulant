import numpy as np
from typing import Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import Operation
from qiskit.circuit.library import QFT
from qiskit.providers import JobV1, Backend


def sample_inner_product(vec_b: np.ndarray, q_pow:int, shots:int=1024) -> Tuple[float, float]:    
    b_prod = np.abs(vec_b) ** 2 
    samples = np.random.choice(b_prod.size, size=shots, p=b_prod)
    num = vec_b[(samples - q_pow) % len(vec_b)]
    dem = vec_b[samples]
    result = np.average(num / dem)
    return np.real(result), np.imag(result)

def true_inner_product(vec_b: np.ndarray, q_pow:int) -> Tuple[float, float]:
    b_conj = np.conj(vec_b)
    b_shift = np.roll(vec_b, q_pow)
    result = np.dot(b_conj, b_shift)
    return np.real(result), np.imag(result)
    

def quantum_inner_product_promise(U_b_gate: Operation, width:int, backend:Backend, q_pow:int, imag: bool=False, shots:int=1024) -> JobV1:
    ancilla = 1
    q_rot = QuantumRegister(width, 'q')
    q_had = QuantumRegister(width + ancilla, 'q')
    c_had = ClassicalRegister(1, 'c')
    qft_gate = QFT(num_qubits=width, inverse=False, name='qft').to_gate()
    rot_cir = QuantumCircuit(q_rot)
    Theta = [(2 * (2 ** i) * q_pow * np.pi) / (2 ** width) for i in range(width)]
    for i in range(width):
        rot_cir.p(Theta[i], q_rot[i])
    rot_cir_gate = rot_cir.to_gate()
    C_rot_cir_gate = rot_cir_gate.control()

    Hadamard_circuit = QuantumCircuit(q_had, c_had)
    Hadamard_circuit.h(q_had[0])
    if imag:
        Hadamard_circuit.s(q_had[0])
    Hadamard_circuit.append(U_b_gate, [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.append(qft_gate, [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.append(C_rot_cir_gate, [q_had[0]] + [q_had[i] for i in range(ancilla, width + ancilla)])
    Hadamard_circuit.h(q_had[0])
    Hadamard_circuit.measure([q_had[0]], [c_had[0]])
    # Transpile the circuit for Hadamard test
    circuit = transpile(Hadamard_circuit, backend)
    job = backend.run(circuit, shots=shots)
    return job

def eval_promise(job: JobV1):
    out_ = job.result()
    count = out_.get_counts()
    new_count = {'0': 0, '1': 0}
    for k in count.keys():
        new_count[k[-1]] += count[k]
    count = new_count
    if count['0'] == 0:
        p0 = 0
        p1 = 1
    elif count['1'] == 0:
        p0 = 1
        p1 = 0
    else:
        shots = sum(list(count.values()))
        p0 = count['0'] / shots
        p1 = count['1'] / shots
    output = p0 - p1
    return output

if __name__ == "__main__":
    print(sample_inner_product(np.arange(16)/np.sqrt(1240),2))
    print(true_inner_product(np.arange(16)/np.sqrt(1240), 2))