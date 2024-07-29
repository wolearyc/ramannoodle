"""Useful utilities (largely linalg for calculating polarizability model)."""

import itertools

import numpy as np
from numpy.typing import NDArray


def find_duplicates(vectors: list[NDArray[np.float64]]) -> NDArray[np.float64] | None:
    """Return duplicate vector in a list or None if no duplicates found."""
    for vector_1, vector_2 in itertools.combinations(vectors, 2):
        if np.isclose(vector_1, vector_2):
            return vector_1
    return None
