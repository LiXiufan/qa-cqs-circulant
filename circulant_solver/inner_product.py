import time
from datetime import datetime

import numpy as np
from typing import Union, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.providers import JobStatus
from circulant_solver.dot_compute import *
from circulant_solver.util import get_backend
import logging

__all__ = [
    "InnerProduct"
]


class InnerProduct():
    r"""Set the inner product class.

    This class records the inner products used for calculating the auxiliary systems W and r.
    In our code implementation, we calculate all inner products at the first step and
    record the values into instances of this class. Then we can obtain the values by indexes.

    Attributes:
        access (str): different access to the backend
        b (Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]]): quantum circuit for preparing b
        term_number (int): number of decomposition terms
        threshold (int): truncated threshold of our algorithm
        shots (int, optional): number of measurements
    """

    def __init__(self, access: str, b: Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]],
                 term_number: int, threshold: int, shots: int = 1024):
        r"""Set the inner product class.

        This class records the inner products used for calculating the auxiliary systems W and r.
        In our code implementation, we calculate all inner products at the first step and
        record the values into instances of this class. Then we can obtain the values by indexes.

        Args:
            access (str): different access to the backend
            b (Union[np.ndarray, QuantumCircuit, Tuple[Dict[int, complex], int]]): quantum circuit for preparing b
            term_number (int): number of decomposition terms
            threshold (int): truncation threshold of our algorithm
            shots (int, optional): number of measurements
        """
        self.access = access
        self.shots = shots
        self.b = b
        self.power = 2 * term_number + 2 * threshold
        self.pos_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.pos_inner_product_imag = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_imag = np.empty(self.power, dtype=np.float64)
        self.non_q = ["true", "sample", "sparse"]
        if self.access not in self.non_q:
            self.backend = get_backend(self.access)
        self._calculate_inner_product()

    def get_inner_product(self, q_pow: int, imag: bool = False):
        r"""Get the value of an inner product.

        Args:
            q_pow (int): the power of permutation matrix
            imag (bool, optional): False: calculate the real part;
                                   True: calculate the imaginary part

        Returns:
            float / int: the value of an inner product
        """
        if q_pow == 0 and not imag:
            return 1
        elif q_pow == 0 and imag:
            return 0
        elif q_pow > 0 and not imag:
            return self.pos_inner_product_real[q_pow - 1]
        elif q_pow > 0 and imag:
            return self.pos_inner_product_imag[q_pow - 1]
        elif q_pow < 0 and not imag:
            return self.neg_inner_product_real[-(q_pow) - 1]
        elif q_pow < 0 and imag:
            return self.neg_inner_product_imag[-(q_pow) - 1]

    def _calculate_inner_product(self):
        r"""Calculate the inner product according to the access.

        If the access is "sparse", calculate the inner product using the sparce matrix estimator;
        If the access is "true", calculate the inner product using the matrix multiplication estimator;
        If the access is "sample", calculate the inner product using sampling and querying estimator;
        Else, calculate the inner product using the Hadamard test with backends provided by Qiskit;
        """
        if self.access == "sparse":
            if not isinstance(self.b, tuple):
                raise NotImplementedError("sparse mode is used with input Tuple[Dict[idx, value], size]")
            dict_b, size = self.b
            for i in range(self.power):
                self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = sparse_inner_product(dict_b, i + 1,
                                                                                                      size)
                self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = sparse_inner_product(dict_b, -(i + 1),
                                                                                                      size)
        elif self.access == "true" or self.access == "sample":
            if isinstance(self.b, np.ndarray):
                vec_b = self.b
            else:
                if isinstance(self.b, QuantumCircuit):
                    sim = Aer.get_backend('unitary_simulator')
                    job = execute(self.b, sim)
                    result = job.result()
                    mat = result.get_unitary(self.b, decimals=16)
                    vec_b = np.transpose(mat)[0]
            if self.access == "true":
                for i in range(self.power):
                    self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = true_inner_product(vec_b, i + 1)
                    self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = true_inner_product(vec_b, -(i + 1))
            elif self.access == "sample":
                for i in range(self.power):
                    self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = sample_inner_product(vec_b, i + 1,
                                                                                                          self.shots)
                    self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = sample_inner_product(vec_b,
                                                                                                          -(i + 1),
                                                                                                          self.shots)
        else:
            if isinstance(self.b, np.ndarray):
                width = int(np.log2(self.b.size))
                q_b = QuantumRegister(width, 'q')
                q_b_cir = QuantumCircuit(q_b)
                U_b = q_b_cir.prepare_state(state=Statevector(self.b)).instructions[0]
            else:
                U_b = self.b.to_gate()
                width = U_b.num_qubits
            promise_queue = []
            pr = []
            pi = []
            nr = []
            ni = []
            for i in range(self.power):
                pos_real = quantum_inner_product_promise(U_b, width, self.backend, i + 1, shots=self.shots, imag=False)
                pos_imag = quantum_inner_product_promise(U_b, width, self.backend, i + 1, shots=self.shots, imag=True)
                neg_real = quantum_inner_product_promise(U_b, width, self.backend, -(i + 1), shots=self.shots,
                                                         imag=False)
                neg_imag = quantum_inner_product_promise(U_b, width, self.backend, -(i + 1), shots=self.shots,
                                                         imag=True)
                promise_queue.append(pos_real)
                promise_queue.append(pos_imag)
                promise_queue.append(neg_real)
                promise_queue.append(neg_imag)
                pr.append(pos_real)
                pi.append(pos_imag)
                nr.append(neg_real)
                ni.append(neg_imag)

            start = datetime.now()
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING,
                                handlers=[logging.FileHandler(f"queue_{start.strftime('%Y%m%d%H%M%S')}.log"),
                                          logging.StreamHandler()])
            logging.warning(f"access: {self.access}, shots: {self.shots}, power:{self.power}")
            time.sleep(self.power * 0.1)
            counter = len(promise_queue)
            while len(promise_queue) > 0:
                job = promise_queue.pop()
                status = job.status()
                counter -= 1
                if status == JobStatus.ERROR:
                    raise RuntimeError("Job failed.")
                elif status == JobStatus.CANCELLED:
                    raise RuntimeError("Job cancelled.")
                elif status == JobStatus.DONE:
                    logging.warning(f'Remaining jobs:{len(promise_queue)}')
                    counter = len(promise_queue)
                else:
                    promise_queue.append(job)
                    if counter == 0:
                        counter = len(promise_queue)
                        logging.warning('Waiting time: {:.2f} hours'.format((datetime.now() - start).seconds / 3600.0))
                        time.sleep(60 * 15)
            logging.warning('Queue cleared; total time: {:.2f} hours'.format((datetime.now() - start).seconds / 3600.0))
            for i in range(self.power):
                self.pos_inner_product_real[i] = eval_promise(pr[i])
                self.pos_inner_product_imag[i] = -eval_promise(pi[i])
                self.neg_inner_product_real[i] = eval_promise(nr[i])
                self.neg_inner_product_imag[i] = -eval_promise(ni[i])


# Test
if __name__ == "__main__":
    print(InnerProduct("true", np.array([1, 1j, -1, -1j]) / 2, 1, 2, 1024).pos_inner_product_imag)
    print(InnerProduct("sample", np.array([1, 1j, -1, -1j]) / 2, 1, 2, 1024).pos_inner_product_imag)
    print(InnerProduct("qiskit-aer", np.array([1, 1j, -1, -1j]) / 2, 1, 2, 1024).pos_inner_product_imag)
