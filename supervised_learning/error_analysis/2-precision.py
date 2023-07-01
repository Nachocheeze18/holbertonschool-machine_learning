#!/usr/bin/env python3
"""precision"""

import numpy as np


def precision(confusion):
    """calculates percision of each class"""
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    
    for i in range(classes):
        true = confusion[i, i]
        predicted = np.sum(confusion[:, i])
        
        if predicted == 0:
            precision[i] = 0
        else:
            precision[i] = true / predicted
    
    return precision