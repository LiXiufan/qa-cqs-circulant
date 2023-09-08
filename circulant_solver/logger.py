import numpy as np
import sys
from circulant_solver.circulant import Circulant
from qiskit import QuantumCircuit, Aer, execute
import json

__all__ = [
    "log"
]

def log(C:Circulant, U_b, W, r, T, alpha, loss, access, shots, log_file):
    np.set_printoptions(threshold=sys.maxsize)
    if isinstance(U_b, QuantumCircuit):
        sim = Aer.get_backend('unitary_simulator')
        job = execute(U_b, sim)
        result = job.result()
        mat = result.get_unitary(U_b, decimals=16)
        vec_b = np.transpose(mat)[0]
        repr_b = str(U_b.draw("latex_source"))
    elif isinstance(U_b, np.ndarray):
        vec_b = U_b
        repr_b = str(vec_b)
    else:
        raise NotImplementedError
    dim = vec_b.size
    c_coeff = dict(zip(C.get_pows(), C.get_coeffs()))
    c_mat = C.get_matrix(dim)
    b_shift = np.zeros((2*T+1, dim), dtype=np.complex128)
    for idx, q_pow in enumerate(range(-T, T+1)):
        b_shift[idx] = np.roll(vec_b, q_pow)
    alpha = np.array(alpha)
    x = np.matmul(alpha, b_shift)
    kappa = np.linalg.cond(c_mat)
    with open(log_file+".txt", 'a') as fp:
        fp.write(f"{c_coeff}-{T}\n\n")
        fp.write(f"C_coeff\n{c_coeff}\n\n")
        fp.write(f"U_b\n{repr_b}\n\n")
        fp.write(f"W\n{str(W)}\n\n")
        fp.write(f"r\n{str(r)}\n\n")
        fp.write(f"Threshold T\n{T}\n\n")
        fp.write(f"alpha\n{str(alpha)}\n\n")
        fp.write(f"x\n{str(x)}\n\n")
        fp.write(f"kappa\n{kappa}\n\n")
        fp.write(f"loss\n{loss}\n\n")
        fp.write(f"access\n{access}\n\n")
        fp.write(f"shots \n{shots}\n\n")
    output = {}
    output["C_coeff"] = c_coeff
    output["U_b"] = repr_b
    output["W"] = str(W)
    output["r"] = str(r)
    output["T"] = T
    output["alpha"] = str(alpha)
    output["x"] = str(x)
    output["kappa"] = kappa
    output["loss"] = loss
    output["access"] = access
    output["shots"] = shots
    out_file = open(log_file+".json", "a")
    json.dump(output, out_file, indent = 6)
    out_file.close()