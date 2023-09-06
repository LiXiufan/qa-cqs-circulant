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
    This is the mitigation module if the bases are Pauli strings.
"""

def Richardson_extrapolate(Shots, Unmiti):
    exp_mtg = 0
    length = len(Shots)
    for m in range(length):
        exp_obs = Unmiti[m]
        shot_m = Shots[m]
        multiplication = 1
        for k in range(length):
            if k == m:
                continue
            else:
                shot_k = Shots[k]
                # multiplication *= (shot_m ** 0.5) / (shot_m ** 0.5 - shot_k ** 0.5)
                multiplication *= (shot_m) / (shot_m - shot_k)

        exp_mtg += exp_obs * multiplication
    return exp_mtg


def Pauli_error_mitigate(p0, p1, file_name='message.txt'):
    # Pauli error mitigation
    if p0 < 0.2:
        p0 = 0
        p1 = 1
        file1 = open(file_name, "a")
        file1.writelines(["p0 < 0.2: set p0 = 0 and p1 = 1.", '\n'])
        file1.writelines(["Expectation value is -1.", '\n'])
        file1.close()
        # print("p0 < 0.2: set p0 = 0 and p1 = 1.")
        # print("Expectation value is -1.")
    else:
        if p1 < 0.2:
            p0 = 1
            p1 = 0
            file1 = open(file_name, "a")
            file1.writelines(["p0 > 0.8: set p0 = 1 and p1 = 0.", '\n'])
            file1.writelines(["Expectation value is 1.", '\n'])
            file1.close()
            # print("p0 > 0.8: set p0 = 1 and p1 = 0.")
            # print("Expectation value is 1.")
        else:
            p0 = 0.5
            p1 = 0.5
            file1 = open(file_name, "a")
            file1.writelines(["0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.", '\n'])
            file1.writelines(["Expectation value is 0.", '\n'])
            file1.close()
            # print("0.2 <= p0 <= 0.8: set p0 = 0.5 and p1 = 0.5.")
            # print("Expectation value is 0.")
    exp = p0 - p1
    return exp
