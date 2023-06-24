#!/usr/bin/env python3
"""normalize"""

import numpy as np


def normalize(X, m, s):
    """stadardize the matrix"""
    norm = (X - m) / s
    
    return norm