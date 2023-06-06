#!/usr/bin/env python3
"""neural network"""


import numpy as np


class NeuralNetwork:
    """Neural"""
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """forward prop"""
        self.__A1 = self.sigmoid(np.dot(self.__W1, X) + self.__b1)
        self.__A2 = self.sigmoid(np.dot(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def sigmoid(self, Z):
        """sigmoid"""
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """cost func"""
        cs = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        """cs = cost"""
        return cs

    def evaluate(self, X, Y):
        """evaluate func"""
        _, A2 = self.forward_prop(X)
        predictions = np.where(A2 >= 0.5, 1, 0)
        cs = self.cost(Y, A2)
        return predictions, cs

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """descent function"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
