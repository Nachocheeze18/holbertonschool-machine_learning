#!/usr/bin/env python3
"""one hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """numeric label to matrix"""
    try:
        m = len(Y)
        encoded = np.zeros((classes, m))
        encoded[Y, np.arange(m)] = 1
        return encoded
    except Exception as e:
        print("Error:", str(e))
        return None
