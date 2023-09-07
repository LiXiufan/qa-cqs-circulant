import numpy as np
from typing import Union
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.providers import JobStatus
from circulant_solver.dot_compute import *
from circulant_solver.util import get_backend

class InnerProduct():
    def __init__(self, access:str, b: Union[np.ndarray, QuantumCircuit], term_number:int, threshold:int, shots:int=1024):
        self.access = access
        self.shots = shots
        self.b = b
        self.power = term_number + threshold - 2
        self.pos_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.pos_inner_product_imag = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_real = np.empty(self.power, dtype=np.float64)
        self.neg_inner_product_imag = np.empty(self.power, dtype=np.float64)
        if not self.access == "true" and not self.access == "sample":
            self.backend = get_backend(self.access)
        self._calculate_inner_product()
    
    def get_inner_product(self, q_pow: int, imag: bool = False):
        if q_pow == 0 and not imag:
            return 1
        elif q_pow == 0 and imag:
            return 0
        elif q_pow > 0 and not imag:
            return self.pos_inner_product_real[q_pow-1]
        elif q_pow > 0 and imag:
            return self.pos_inner_product_imag[q_pow-1]
        elif q_pow < 0 and not imag:
            return self.neg_inner_product_real[-(q_pow)-1]
        elif q_pow < 0 and imag:
            return self.neg_inner_product_imag[-(q_pow)-1]
        
    def _calculate_inner_product(self):
        if self.access == "true" or self.access == "sample":
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
                    self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = true_inner_product(vec_b, i+1)
                    self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = true_inner_product(vec_b, -(i+1))
            elif self.access == "sample":
                for i in range(self.power):
                    self.pos_inner_product_real[i], self.pos_inner_product_imag[i] = sample_inner_product(vec_b, i+1, self.shots)
                    self.neg_inner_product_real[i], self.neg_inner_product_imag[i] = sample_inner_product(vec_b, -(i+1), self.shots)
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
                pos_real = quantum_inner_product_promise(U_b, width, self.backend, i+1, shots=self.shots, imag=False)
                pos_imag = quantum_inner_product_promise(U_b, width, self.backend, i+1, shots=self.shots, imag=True)
                neg_real = quantum_inner_product_promise(U_b, width, self.backend, -(i+1), shots=self.shots, imag=False)
                neg_imag = quantum_inner_product_promise(U_b, width, self.backend, -(i+1), shots=self.shots, imag=True)
                promise_queue.append(pos_real)
                promise_queue.append(pos_imag)
                promise_queue.append(neg_real)
                promise_queue.append(neg_imag)
                pr.append(pos_real)
                pi.append(pos_imag)
                nr.append(neg_real)
                ni.append(neg_imag)
            while len(promise_queue) > 0:
                job = promise_queue.pop()
                status = job.status()
                if status == JobStatus.ERROR:
                    raise RuntimeError("Job failed.")
                elif status == JobStatus.CANCELLED:
                    raise RuntimeError("Job cancelled.")
                elif status != JobStatus.DONE:
                    promise_queue.append(job)
            for i in range(self.power):
                self.pos_inner_product_real[i] = eval_promise(pr[i])
                self.pos_inner_product_imag[i] = -eval_promise(pi[i])
                self.neg_inner_product_real[i] = eval_promise(nr[i])
                self.neg_inner_product_imag[i] = -eval_promise(ni[i])

if __name__ == "__main__":
    print(InnerProduct("true", np.array([1, 1j, -1, -1j])/2, 1, 2, 1024).pos_inner_product_imag)
    print(InnerProduct("sample", np.array([1, 1j, -1, -1j])/2, 1, 2, 1024).pos_inner_product_imag)
    print(InnerProduct("qiskit-aer", np.array([1, 1j, -1, -1j])/2, 1, 2, 1024).pos_inner_product_imag)