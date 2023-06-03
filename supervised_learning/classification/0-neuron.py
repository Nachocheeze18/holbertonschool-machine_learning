#!/usr/bin/env python3
"""Neuron"""
import numpy as np

class Neuron:
    def __init__(self, nx):
        self.nx = nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(nx,))
        self.b = 0
        self.A = 0

