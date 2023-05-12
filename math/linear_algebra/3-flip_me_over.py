#!/usr/bin/env python3
"""code to transpose a matrix"""


def matrix_transpose(matrix):
    """return transpose of a 2D matrix"""
    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[0 for j in range(rows)] for i in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]

    return transpose
