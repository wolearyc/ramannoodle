"""Useful utilities (largely linalg for calculating polarizability model)"""

import numpy as np
from numpy.typing import NDArray


def are_collinear(vector_1: NDArray[np.float64], vector_2: NDArray[np.float64]) -> bool:
    """Checks if two vectors are collinear"""
    vector_1 /= np.linalg.norm(vector_1)
    vector_2 /= np.linalg.norm(vector_2)
    dot_product = vector_1.dot(vector_2)
    result: bool = np.isclose(dot_product, 1).all() or np.isclose(dot_product, -1).all()
    return result


def check_orthogonal(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Checks if a vector is orthogonal to all vectors in a list. Returns
    first index of non-orthogonal vector, otherwise returns -1"""

    # This implementation could be made more efficient but readability would
    # be sacrificed .
    for index, vector_2 in enumerate(vectors):
        if np.dot(vector_1, vector_2) ** 2 > 0.001:
            return index

    return -1
