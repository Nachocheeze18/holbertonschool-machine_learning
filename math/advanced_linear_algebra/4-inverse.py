#!/usr/bin/env python3
"""inverse matrix"""

def determinant(submatrix):
    if len(submatrix) == 1:
        return submatrix[0][0]
    elif len(submatrix) == 2:
        return (submatrix[0][0]
                * submatrix[1][1]
                - submatrix[0][1]
                * submatrix[1][0])

    det = 0
    for j in range(len(submatrix)):
        cofactor_sign = (-1) ** j
        cofactor_value = submatrix[0][j]
        submatrix_minor = [row[:j] + row[j + 1:] for row in submatrix[1:]]
        det += (cofactor_sign * cofactor_value *
                determinant(submatrix_minor))
        return det

def inverse(matrix):
    """Calculates the inverse matrix of a given square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    num_rows = len(matrix)
    if not all(len(row) == num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    determinant_value = determinant(matrix)

    if determinant_value == 0:
        raise ValueError("Matrix is singular; it does not have an inverse.")

    adjugate_matrix = adjugate(matrix)

    inverse_matrix = [[adjugate_matrix[i][j] / determinant_value
                       for j in range(num_rows)] for i in range(num_rows)]

    return inverse_matrix

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

    if num_rows == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(num_rows):
        cofactor_row = []
        for j in range(num_rows):
            submatrix = [row[:j] + row[j + 1:] for row in (matrix[:i] +
                                                           matrix[i + 1:])]
            cofactor_sign = (-1) ** (i + j)
            cofactor_value = cofactor_sign * determinant(submatrix)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculates the adjugate matrix of a given matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    num_rows = len(matrix)
    if not all(len(row) == num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = []
    for i in range(num_rows):
        adjugate_row = []
        for j in range(num_rows):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
