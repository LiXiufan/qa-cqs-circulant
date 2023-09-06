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
    This is the calculator module to compute Q and r according to the Hadamard test shooting outcomes.
"""

from numpy import array
from numpy import zeros
from numpy import append

from numpy import real, imag
from numpy import conj

from hardware.execute import Hadamard_test_QFT
import time
from datetime import datetime
from cqs.mitigation import Pauli_error_mitigate


def calculate_statistics(backend, jobs_ids, Pauli_mitigate=False, file_name='message.txt'):
    # a list of jobs
    exps = []
    for job_id in jobs_ids:
        if job_id == 1 or job_id == 0:
            exp = job_id
            # print("This circuit is composed of identities, skip.")
            file1 = open(file_name, "a")
            file1.writelines(["This circuit is composed of identities, skip.\n"])
            file1.close()
        else:
            job = backend.retrieve_job(job_id)
            status = job.status()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            file1 = open(file_name, "a")
            file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
            file1.writelines(["Current Status:", str(status), '\n\n'])
            file1.close()
            print("Current Time =", current_time)
            print('Current Status:', status)
            print()
            DONE = status.DONE
            while status != DONE:
                time.sleep(3600)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                status = job.status()
                file1 = open(file_name, "a")
                file1.writelines(["\nCurrent Time =", str(current_time), '\n'])
                file1.writelines(["Current Status:", str(status), '\n\n'])
                file1.close()
                print("Current Time =", current_time)
                print('Status:', status)
                print()
            count = backend.retrieve_job(job_id).result().get_counts()
            new_count = {'0': 0, '1': 0}
            for k in count.keys():
                new_count[k[-1]] += count[k]
            count = new_count
            file1 = open(file_name, "a")
            file1.writelines(["The sampling result is:", str(count), '\n'])
            file1.close()
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

            if Pauli_mitigate is True:
                exp = Pauli_error_mitigate(p0, p1, file_name)
            else:
                exp = p0 - p1
        exps.append(exp)
    return exps


def calculate_W_r(C, U_b, Ansatz_pows, access=None, file_name='message.txt'):
    r"""
        Calculate the auxiliary system W and r defined in our paper.
    """
    C_coeffs = C.get_coeffs()
    C_pows = C.get_pows()
    K = len(C_coeffs)
    T = len(Ansatz_pows)
    V_dagger_V = zeros((T, T), dtype='complex128')

    Job_ids_K_R = []
    Job_ids_K_I = []
    Job_ids_q_R = []
    Job_ids_q_I = []
    for t_1 in range(T):
        for t_2 in range(T):
            # Uniform distribution of the shots
            for k_1 in range(K):
                for k_2 in range(K):
                    q_pow = - Ansatz_pows[t_1] - C_pows[k_1] + C_pows[k_2] + Ansatz_pows[t_2]
                    backend, jobid_R = Hadamard_test_QFT(U_b, q_pow, alpha=1, access=access)
                    backend, jobid_I = Hadamard_test_QFT(U_b, q_pow, alpha=1j, access=access)
                    Job_ids_K_R.append(jobid_R)
                    Job_ids_K_I.append(jobid_I)
    for t in range(T):
        for k in range(K):
            q_pow = Ansatz_pows[t] + C_pows[k]
            backend, jobid_R = Hadamard_test_QFT(U_b, q_pow, alpha=1, access=access)
            backend, jobid_I = Hadamard_test_QFT(U_b, q_pow, alpha=1j, access=access)
            Job_ids_K_R.append(jobid_R)
            Job_ids_K_I.append(jobid_I)

    exp_K_R = calculate_statistics(backend, Job_ids_K_R, file_name=file_name)
    exp_K_I = calculate_statistics(backend, Job_ids_K_I, file_name=file_name)
    exp_q_R = calculate_statistics(backend, Job_ids_q_R, file_name=file_name)
    exp_q_I = calculate_statistics(backend, Job_ids_q_I, file_name=file_name)

    for t_1 in range(T):
        for t_2 in range(T):
            # Uniform distribution of the shots
            item = 0
            for k_1 in range(K):
                for k_2 in range(K):
                    inner_product_real = exp_K_R[t_1 * T * K * K +
                                                 t_2 * K * K +
                                                 k_1 * K +
                                                 k_2]
                    inner_product_imag = exp_K_I[t_1 * T * K * K +
                                                 t_2 * K * K +
                                                 k_1 * K +
                                                 k_2]
                    inner_product = inner_product_real - inner_product_imag * 1j
                    item += conj(C_coeffs[k_1]) * C_coeffs[k_2] * inner_product
            V_dagger_V[t_1][t_2] = item

    R = real(V_dagger_V)
    I = imag(V_dagger_V)

    q = zeros((T, 1), dtype='complex128')
    for t in range(T):
        item = 0
        for k in range(K):
            inner_product_real = exp_q_R[t * K + k]
            inner_product_imag = exp_q_I[t * K + k]
            inner_product = inner_product_real - inner_product_imag * 1j
            item += C_coeffs[k] * inner_product
        q[t][0] = item

    # W     =      R    -I
    #       =      I     R
    W = array(append(append(R, -I, axis=1), append(I, R, axis=1), axis=0), dtype='float64')

    # r = [Re(q),
    #      Im(q)]
    r_real = real(q)
    r_imag = imag(q)
    r = array(append(r_real, r_imag, axis=0), dtype='float64')
    return W, r

########################################################################################################################
