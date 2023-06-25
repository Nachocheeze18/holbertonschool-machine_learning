#!/usr/bin/env python3
"""moving average"""

import numpy as np


def moving_average(data, beta):
    """weighted fomula"""
    x = 0
    x_List = []
    for y, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError("Data points must be numeric.")
        
        x = ((x * beta) + ((1 - beta) * value))
        b = x / (1 - (beta ** (y + 1)))
        x_List.append(b)
    return x_List
