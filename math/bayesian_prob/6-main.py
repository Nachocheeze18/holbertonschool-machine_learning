#!/usr/bin/env python3

intersection = __import__('1-intersection').intersection
import numpy as np

try:
    intersection(25, 30, np.linspace(0, 1, 11), np.array([1, -1, 0.9, -0.9, 0.5, -0.5, 0.2, 0.2, 0.2, 0.2, 0.2]))
except ValueError as e:
    print(str(e))
try:
    intersection(25, 30, np.linspace(0, 1, 11), np.array([1.1, -1.1, 1, -1, 0.5, -0.5, 0.2, 0.2, 0.2, 0.2, 0.2]))
except ValueError as e:
    print(str(e))