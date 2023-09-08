import numpy as np
from circulant_solver.inner_product import InnerProduct

__all__ = [
    "calculate_W_r"
]


def calculate_W_r(C, b, Ansatz_pows, access=None, shots=1024):
    r"""
        Calculate the auxiliary system W and r defined in our paper.
    """
    C_coeffs = C.get_coeffs()
    C_pows = C.get_pows()
    K = len(C_coeffs)
    T = len(Ansatz_pows)
    V_R = np.zeros((T, T), dtype=np.float64)
    V_I = np.zeros((T, T), dtype=np.float64)
    q_R = np.zeros((T, 1), dtype=np.float64)
    q_I = np.zeros((T, 1), dtype=np.float64)
    ip = InnerProduct(access, b, K, T, shots)
    for t_1 in range(T):
        for t_2 in range(T):
            for k_1 in range(K):
                for k_2 in range(K):
                    q_pow = - Ansatz_pows[t_1] - C_pows[k_1] + C_pows[k_2] + Ansatz_pows[t_2]
                    V_R[t_1][t_2] += np.conj(C_coeffs[k_1]) * C_coeffs[k_2] * ip.get_inner_product(q_pow, imag=False)
                    V_I[t_1][t_2] += np.conj(C_coeffs[k_1]) * C_coeffs[k_2] * ip.get_inner_product(q_pow, imag=True)
    for t in range(T):
        for k in range(K):
            q_pow = Ansatz_pows[t] + C_pows[k]
            q_R[t] += C_coeffs[k] * ip.get_inner_product(q_pow, imag=False)
            q_I[t] += C_coeffs[k] * ip.get_inner_product(q_pow, imag=True)
    W = np.array(np.append(np.append(V_R, -V_I, axis=1), np.append(V_I, V_R, axis=1), axis=0), dtype='float64')
    r = np.array(np.append(q_R, q_I, axis=0), dtype='float64')
    return W, r
