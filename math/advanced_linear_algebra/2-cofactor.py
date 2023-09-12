#!/usr/bin/env python3
"""cofactor matrix"""


def cofactor(matrix):
    """calculates the cofactor matrix of a given square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    num_rows = len(matrix)
    if not all(len(row) == num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        elif len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        det = 0
        for j in range(len(matrix)):
            cofactor_sign = (-1) ** j
            cofactor_value = matrix[0][j]
            submatrix_minor = [row[:j] + row[j + 1:] for row in matrix[1:]]
            det += cofactor_sign * cofactor_value * determinant
            (submatrix_minor)
        return det

    if num_rows == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(num_rows):
        cofactor_row = []
        for j in range(num_rows):
            submatrix = [row[:j] + row[j + 1:] for row in
                         (matrix[:i] + matrix[i + 1:])]
            cofactor_sign = (-1) ** (i + j)
            cofactor_value = cofactor_sign * determinant(submatrix)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
