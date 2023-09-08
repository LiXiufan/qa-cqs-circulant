import numpy as np
from qiskit import Aer

__all__ = [
    "get_permutation_matrix",
    "get_backend"
]


def get_permutation_matrix(dim: int, p: int):
    r"""Generate the matrix of permutation operator.

    Args:
        dim (int): dimension
        p (int): power
    """
    Q = np.zeros((dim, dim))
    for k in range(0, dim):
        Q[k][k - p] = 1
    return Q


def get_backend(access):
    if access == 'qiskit-aer':
        backend = Aer.get_backend('aer_simulator_statevector')
    elif access == 'ibmq-statevector':
        try:
            from qiskit_ibm_provider import IBMProvider
        except ImportError:
            raise ImportError("Please pip install qiskit-ibm-provider for this option.")
        provider = IBMProvider(instance='ibm-q/open/main')
        hub = "ibm-q"
        group = "open"
        project = "main"
        backend_name = "simulator_statevector"
        backend = provider.get_backend(backend_name, instance=f"{hub}/{group}/{project}")

    elif access == 'ibmq-perth':
        try:
            from qiskit_ibm_provider import IBMProvider
        except ImportError:
            raise ImportError("Please pip install qiskit-ibm-provider for this option.")
        provider = IBMProvider(instance='ibm-q/open/main')
        hub = "ibm-q"
        group = "open"
        project = "main"
        backend_name = "ibm_perth"
        backend = provider.get_backend(backend_name, instance=f"{hub}/{group}/{project}")

    else:
        raise NotImplementedError

    return backend
