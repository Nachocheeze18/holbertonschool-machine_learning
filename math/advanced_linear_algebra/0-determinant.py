#!/usr/bin/env python3
"""Matrix Determinant Calculator"""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.
    """

    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        for _ in range(3):
            print("matrix must be a list of lists")
        return

    n = len(matrix)

    if n == 0:
        return 1

    if n != len(matrix[0]):
        for _ in range(3):
            print("matrix must be a square matrix")
        return

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    result = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        cofactor = matrix[0][j] * determinant(sub_matrix)
        if cofactor is None:
            return None
        result += cofactor if j % 2 == 0 else -cofactor

    return result
