#!/usr/bin/env python3
"""specificity """

import numpy as np


def specificity(confusion):
    """calculates the specificity for each class """
    classes = confusion.shape[0]
    specificity_arr = np.zeros(classes)

    for i in range(classes):
        t_negative = np.sum(np.delete(np.delete(confusion, i, axis=0),
                                      i, axis=1))
        f_positive = np.sum(confusion[:, i]) - confusion[i, i]
        specificity = t_negative / (t_negative + f_positive)
        specificity_arr[i] = specificity

    return specificity_arr
