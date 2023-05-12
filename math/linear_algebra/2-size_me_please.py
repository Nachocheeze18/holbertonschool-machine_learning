#!/usr/bin/env python3
"""code to calculate shapes"""


def matrix_shape(matrix):
    """calculate the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
