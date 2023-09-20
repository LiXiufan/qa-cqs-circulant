import numpy as np
from typing import Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import Operation
from qiskit.circuit.library import QFT
from qiskit.providers import JobV1, Backend

__all__ = [
    "sample_inner_product",
    "true_inner_product",
    "sparse_inner_product",
    "quantum_inner_product_promise",
    "eval_promise"
]


def sample_inner_product(vec_b: np.ndarray, q_pow: int, shots: int = 1024) -> Tuple[float, float]:
    r"""Estimate the inner products by sampling and querying.

    Args:
        vec_b (np.ndarray): vector b
        q_pow (int): the power of permutation matrix
        shots (int, optional): number of measurements

    Returns:
        Tuple[float, float]: real and imaginary part of the inner product
    """
    b_prod = np.abs(vec_b) ** 2
    samples = np.random.choice(b_prod.size, size=shots, p=b_prod)
    shift = (samples - q_pow) % vec_b.size
    num = vec_b[shift]
    dem = vec_b[samples]
    result = np.average(num / dem)
    return np.real(result), np.imag(result)


def true_inner_product(vec_b: np.ndarray, q_pow: int) -> Tuple[float, float]:
    r"""Estimate the inner products by matrix multiplication.

    Args:
        vec_b (np.ndarray): vector b
        q_pow (int): the power of permutation matrix

    Returns:
        Tuple[float, float]: real and imaginary part of the inner product
    """
    b_conj = np.conj(vec_b)
    b_shift = np.roll(vec_b, q_pow)
    result = np.dot(b_conj, b_shift)
    return np.real(result), np.imag(result)


def sparse_inner_product(dict_b: Dict[int, complex], q_pow: int, size: int) -> Tuple[float, float]:
    r"""Estimate the inner products by simple shifting the elements.

    Args:
        dict_b (Dict[int, complex]): vector b
        q_pow (int): the power of permutation matrix
        size (int): the size of the matrix

    Returns:
        Tuple[float, float]: real and imaginary part of the inner product
    """
    result = 0
    for idx, value in dict_b.items():
        shifted_idx = idx - q_pow
        if shifted_idx >= size:
            shifted_idx -= size
        if shifted_idx < -size:
            shifted_idx += size
        if shifted_idx in dict_b:
            result += np.conj(value) * dict_b[shifted_idx]
    return np.real(result), np.imag(result)


def quantum_inner_product_promise(U_b_gate: Operation, width: int, backend: Backend, q_pow: int, imag: bool = False,
                                  shots: int = 1024) -> JobV1:
    r"""Estimate the inner products by Hadamard test.

    Args:
        U_b_gate (Operation): the unitary circuit used to prepare the vector b
        width (int): width of the circuit
        backend (Backend): the backend supported on Qiskit
        q_pow (int): the power of permutation matrix
        imag (bool, optional): False: calculate the real part;
                               True: calculate the imaginary part
        shots (int, optional): number of measurements

    Returns:
        JobV1: submitted job corresponding to the Hadamard test task
    """
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


def eval_promise(job: JobV1) -> float:
    r"""Retrieve the results of submitted job.

    The waiting list might be extremely long in terms of real hardware experiments.
    To improve the efficiency, the user can submit the jobs in parallel and retrieve the results later.

    Args:
        job (JobV1): submitted job corresponding to the Hadamard test task

    Returns:
        float: the estimation of the inner product by statistics
    """
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


# Test
if __name__ == "__main__":
    print(sample_inner_product(np.arange(16) / np.sqrt(1240), 2))
    print(true_inner_product(np.arange(16) / np.sqrt(1240), 2))
