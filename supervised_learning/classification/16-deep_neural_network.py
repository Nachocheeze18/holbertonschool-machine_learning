#!/usr/bin/env python3
"""deep neural network"""
import numpy as np

class DeepNeuralNetwork:
    """Deep Neural Network"""
    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for l in layers:
            if not isinstance(l, int) or l < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            self.__weights['W' + str(l)] = np.random.randn(layers[l-1], nx) * np.sqrt(2 / nx)
            self.__weights['b' + str(l)] = np.zeros((layers[l-1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
