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

    def evaluate(self, X, Y):
        """Evaluates the network's predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        L = self.__L

        A = cache["A" + str(L)]
        dZ = A - Y

        for l in reversed(range(1, L+1)):
            A_prev = cache["A" + str(l - 1)]
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db

            if l > 1:
                dZ = dA * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

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
