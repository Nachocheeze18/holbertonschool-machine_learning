#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network"""
    def __init__(self, nx, layers):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1 or False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, layer_size in enumerate(layers):
            if i == 0:
                input_size = nx
            else:
                input_size = layers[i-1]

            self.__weights['W' + str(i+1)] = (
                np.random.randn(layer_size, input_size) *
                np.sqrt(2 / input_size)
            )
            self.__weights['b' + str(i+1)] = np.zeros((layer_size, 1))

    def forward_prop(self, X):
        """Performs forward propagation"""
        self.__cache['A0'] = X

        for l in range(1, self.__L + 1):
            A_prev = self.__cache['A' + str(l-1)]
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache['A' + str(l)] = A

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """model cost"""
        m = Y.shape[1]
        cs = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        """cs = cost"""
        return cs

    @property
    def L(self):
        """Getter for L (number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

