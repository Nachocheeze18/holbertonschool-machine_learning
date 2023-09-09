#!/usr/bin/env python3
"""Matrix Determinant Calculator"""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 0:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    result = 0
    i = 0
    while i < len(matrix):
        sub = [[matrix[j][k] for k in range(len(matrix[j])) if k != i] for j in range(1, len(matrix))]
        cofactor = matrix[0][i] * determinant(sub)
        if i % 2 == 0:
            result += cofactor
        else:
            result -= cofactor
        i += 1

    return result
