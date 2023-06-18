#!/usr/bin/env python3
"""one hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """matrix to vectors"""
    if not isinstance(one_hot, np.ndarray):
        return None

    if one_hot.ndim != 2:
        return None

    return np.argmax(one_hot, axis=0)
