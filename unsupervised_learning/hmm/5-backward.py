#!/usr/bin/env python3
"""Imports"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    T = len(Observation)
    N, M = Emission.shape

    if T == 0 or N == 0 or M == 0:
        return None, None

    B = np.zeros((N, T))
    for i in range(N):
        B[i, T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                B[i, t] += B[j, t + 1] * Transition[i, j] * Emission[j,
                                                                     Observation[t + 1]]

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
