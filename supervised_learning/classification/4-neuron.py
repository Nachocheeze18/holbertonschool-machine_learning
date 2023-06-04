#!/usr/bin/env python3
"""nueron"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Cunstructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """Forward Propogation"""
        self.__A = self.sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def sigmoid(self, X):
        """Sigmoid"""
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        """model cost"""
        m = Y.shape[1]
        cs = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        """cs = cost"""
        return cs

    def evaluate(self, X, Y):
        """Evaluate func"""
        m = Y.shape[1]
        prediction = np.zeros((1, m))

        A = self.forward_prop(X)
        prediction[A >= 0.5] = 1

        return prediction, self.cost(Y, A)
    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A