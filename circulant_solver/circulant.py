import numpy as np
from circulant_solver.util import get_permutation_matrix

__all__ = [
    "Circulant"
]


class Circulant:
    r"""Set the ``C`` circulant matrix.

    This class generates the circulant matrix C of the linear system of equations with a specific forms.
    It returns the permutation orders, coefficients, and matrix.
    Users can also customize the C matrix with specific input.

    We assume the circulant matrix has the linear combination of permutations:

    .. math::

            C = \sum_{m=0}^{N-1} c_m Q^m,

    Attributes:
        term_number (int): number of decomposition terms
    """

    def __init__(self, term_number, permu_pows, coeffs):
        r"""Set the ``C`` circulant matrix.

        This class generates the circulant matrix C of the linear system of equations with a specific forms.
        It returns the permutation orders, coefficients, and matrix.
        Users can also customize the C matrix with specific input.

        Args:
            term_number (int): number of decomposition terms
            permu_pows (list): a list of integers representing different powers of the permutations
            coeffs (list): a list of complex numbers representing different coefficients
        """
        self.__term_number = term_number
        self.__pows = permu_pows
        self.__coeffs = coeffs

    def get_pows(self):
        return self.__pows

    def get_coeffs(self):
        return self.__coeffs

    def get_matrix(self, dim):
        mat = np.array([[0 for _ in range(dim)] for _ in range(dim)], dtype='complex128')
        for i in range(self.__term_number):
            coeff = self.__coeffs[i]
            power = self.__pows[i]
            q_mat = get_permutation_matrix(dim, power)
            mat += coeff * q_mat
        return mat
