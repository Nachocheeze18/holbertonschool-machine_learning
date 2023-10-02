#!/usr/bin/env pyhon3
"""Imports"""
import numpy as np

def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden
    states for a hidden markov model"""
    T = len(Observation)
    N, M = Emission.shape

    if len(Initial.shape) == 2:
        Initial = Initial.reshape(-1)

    V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    epsilon = 1e-10

    V[:, 0] = np.log(Initial + epsilon) + np.log(Emission[:, Observation[0]] + epsilon)

    for t in range(1, T):
        for s in range(N):
            probabilities = V[:, t-1] + np.log(Transition[:, s] + epsilon) + np.log(Emission[s, Observation[t]] + epsilon)

            max_prob_index = np.argmax(probabilities)

            V[s, t] = probabilities[max_prob_index]

            backpointer[s, t] = max_prob_index

    max_prob_index = np.argmax(V[:, -1])
    P = np.exp(V[max_prob_index, -1])

    path = [max_prob_index]
    for t in range(T - 1, 0, -1):
        max_prob_index = backpointer[max_prob_index, t]
        path.insert(0, max_prob_index)

    return path, P
