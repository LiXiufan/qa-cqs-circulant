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
    This is the verifier module to verify the results of Hadamard tests and gradient overlaps
"""

from numpy import array, sqrt, identity
from numpy import kron, conj, transpose
from numpy import real, imag
from numpy import zeros, linalg
from numpy import cos, sin

def I_mat():
    return array([[1, 0], [0, 1]])


def X_mat():
    return array([[0, 1], [1, 0]])


def Y_mat():
    return array([[0, -1j], [1j, 0]])


def Z_mat():
    return array([[1, 0], [0, -1]])


def H_mat():
    return 1 / sqrt(2) * array([[1, 1], [1, -1]])


def R_mat(axis, theta):
    r"""Rotation gate matrix.

    .. math::

        R_{x}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) X

        R_{y}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Y

        R_{z}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Z

    Args:
        axis (str): the rotation axis, 'x', 'y, or 'z'
        theta (float / ndarray): rotation angle

    Returns:
        ndarray: rotation gate matrix
    """
    # Calculate half of the input theta
    if axis == 'x':  # define the rotation - x gate matrix
        return cos(0.5 * theta) * I_mat() - 1j * sin(0.5 * theta) * X_mat()
    elif axis == 'y':  # define the rotation - x gate matrix
        return cos(0.5 * theta) * I_mat() - 1j * sin(0.5 * theta) * Y_mat()
    elif axis == 'z':  # define the rotation - x gate matrix
        return cos(0.5 * theta) * I_mat() - 1j * sin(0.5 * theta) * Z_mat()
    else:
        raise KeyError("invalid rotation gate index: %s, the rotation axis must be 'x', 'y' or 'z'." % axis)


def zero_state():
    return array([1, 0]).reshape(2, 1)


def one_state():
    return array([0, 1]).reshape(2, 1)


def plus_state():
    return (1 / sqrt(2)) * array([1, 1]).reshape(2, 1)


def minus_state():
    return (1 / sqrt(2)) * array([1, -1]).reshape(2, 1)

def get_permutation_matrix(n, p):
    r"""Generate the matrix of permutation operator.

    Args:
        n (int): qubit number
        p (int): power
    """
    dim = 2 ** n
    Q = zeros((dim, dim))
    Q[0][-1] = 1
    for k in range(1, dim):
        Q[k][k - 1] = 1
    Q = linalg.matrix_power(Q, p)
    return Q

def gate_to_matrix(gate):
    name = gate[0]
    if name == 'I':
        u = I_mat()
    elif name == 'X':
        u = X_mat()
    elif name == 'Y':
        u = Y_mat()
    elif name == 'Z':
        u = Z_mat()
    elif name == 'H':
        u = H_mat()
    elif name == 'Rx':
        theta = gate[1]
        u = R_mat('x', theta)
    elif name == 'Ry':
        theta = gate[1]
        u = R_mat('y', theta)
    elif name == 'Rz':
        theta = gate[1]
        u = R_mat('z', theta)
    else:
        raise ValueError
    return u


def get_x(vars, ansatz_tree):
    m = len(vars)
    x = 0
    for i in range(m):
        var = vars[i]
        U = ansatz_tree[i]
        # Matrix calculations
        width = len(U[0])
        zeros = zero_state()
        if width > 1:
            for j in range(width - 1):
                zeros = kron(zeros, zero_state())

        U_mat = identity(2 ** width)
        for layer in U:
            U_layer = array([1])
            for j, gate in enumerate(layer):
                u = gate_to_matrix(gate)
                U_layer = kron(U_layer, u)
            U_mat = U_mat @ U_layer
        x += var * U_mat @ zeros
    return x


def get_unitary(U):
    width = len(U[0])

    # Matrix calculations
    U_mat = identity(2 ** width)
    if len(U[0]) == width:
        for layer in U:
            U_layer = array([1])
            for i, gate in enumerate(layer):
                u = gate_to_matrix(gate)
                U_layer = kron(u, U_layer)
            U_mat = U_mat @ U_layer
    return U_mat


def verify_Hadamard_test_result(hardmard_result, U, alpha=1):
    U_mat = get_unitary(U)

    if alpha == 1:
        ideal = real(
            kron(conj(transpose(zero_state())), conj(transpose(zero_state()))) @
            U_mat @ kron(zero_state(), zero_state())).item()
    elif alpha == 1j:
        ideal = imag(
            kron(conj(transpose(zero_state())), conj(transpose(zero_state()))) @
            U_mat @ kron(zero_state(), zero_state())).item()
    else:
        raise ValueError
    # print('The ideal result is:', ideal)

    error = linalg.norm(ideal - hardmard_result)
    print('The error between the Hadamard test result and the ideal result is:', error)



def verify_loss_function(A, vars, ansatz_tree, loss_es):
    A_mat = A.get_matrix()
    A_unitaries = A.get_unitary()

    x = get_x(vars, ansatz_tree)
    zeros = zero_state()
    width = len(A_unitaries[0][0])
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

    loss = real((conj(transpose(x)) @ conj(transpose(A_mat)) @ A_mat @ x - 2 * real(
        conj(transpose(zeros)) @ A_mat @ x)).item()) + 1
    print("Loss calculated by matrices:", loss)
    error = linalg.norm(loss_es - loss)
    return error


def verify_gradient_overlap(A, vars, ansatz_tree, max_index_es):
    A_mat = A.get_matrix()
    A_coeffs = A.get_coeff()
    A_unitaries = A.get_unitary()
    A_terms_number = len(A_coeffs)

    x = get_x(vars, ansatz_tree)
    zeros = zero_state()
    width = len(A_unitaries[0][0])
    if width > 1:
        for j in range(width - 1):
            zeros = kron(zeros, zero_state())

    parent_node = ansatz_tree[-1]
    child_space = [parent_node + A_unitaries[i] for i in range(A_terms_number)]
    gradient_overlaps = [0 for _ in range(len(child_space))]

    for i, child_node in enumerate(child_space):
        U_mat = get_unitary(child_node)

        gradient = 2 * conj(transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ A_mat @ x - 2 * conj(
            transpose(zeros)) @ conj(transpose(U_mat)) @ A_mat @ zeros
        gradient_overlaps[i] = abs(gradient.item())

    # print('Matrix Calculation:', gradient_overlaps)

    max_index = [index for index, item in enumerate(gradient_overlaps) if item == max(gradient_overlaps)]

    if max_index == max_index_es:
        print("Correct!")

    else:
        raise ValueError("Expansion Module is Wrong!")
