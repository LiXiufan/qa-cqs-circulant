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
    Generator of the circulant linear systems of equations problem.
"""

from numpy import array
from cqs.verifier import get_permutation_matrix

__all__ = [
    "Circulant"
]

ModuleErrorCode = 1
FileErrorCode = 0


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
        dim (int): dimension of the system matrix
        width (int): qubit number
    """
    def __init__(self, term_number, dim, width):
        r"""Set the ``C`` circulant matrix.

        This class generates the circulant matrix C of the linear system of equations with a specific forms.
        It returns the permutation orders, coefficients, and matrix.
        Users can also customize the C matrix with specific input.

        Args:
            term_number (int): number of decomposition terms
            dim (int): dimension of the system matrix
            width (int): qubit number
        """
        self.__pows = None  # orders of permutation matrix
        self.__matrix = None  # matrix
        self.__coeffs = None  # coefficients of the unitaries
        self.__term_number = term_number
        self.__dim = dim
        # self.__width = int(log2(self.__dim))
        self.__width = width

    def generate(self, permu_pows, coeffs):
        r"""Automatically generate a matrix with the given parameters.

        Args:
            permu_pows (list): a list of integers representing different powers of the permutations
            coeffs (list): a list of complex numbers representing different coefficients
        """
        self.__pows = permu_pows
        self.__coeffs = coeffs

    def get_pows(self):
        return self.__pows

    def get_coeffs(self):
        return self.__coeffs

    def get_width(self):
        return self.__width

    def get_matrix(self):
        mat = array([[0 for _ in range(self.__dim)] for _ in range(self.__dim)], dtype='complex128')

        for i in range(self.__term_number):
            coeff = self.__coeffs[i]
            pow = self.__pows[i]
            q_mat = get_permutation_matrix(self.__width, pow)
            mat += coeff * q_mat

        self.__matrix = mat
        return self.__matrix

