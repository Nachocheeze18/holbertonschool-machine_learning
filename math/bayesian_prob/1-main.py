#!/usr/bin/env python3

intersection = __import__('1-intersection').intersection
import numpy as np

try:
    intersection(1, '50', np.linspace(0, 1, 11), np.ones(11) / 11)
except ValueError as e:
    print(str(e))
try:
    intersection(1, -5, np.linspace(0, 1, 11), np.ones(11) / 11)
except ValueError as e:
    print(str(e))
try:
    intersection(0, 0, np.linspace(0, 1, 11), np.ones(11) / 11)
except ValueError as e:
    print(str(e))
