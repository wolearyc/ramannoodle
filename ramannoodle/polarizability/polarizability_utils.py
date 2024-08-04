"""Useful utilities (largely linalg for calculating polarizability model)."""

import itertools
from typing import Iterable

import numpy as np
from numpy.typing import NDArray, ArrayLike


def find_duplicates(vectors: Iterable[ArrayLike]) -> NDArray | None:
    """Return duplicate vector in a list or None if no duplicates found."""
    try:
        combinations = itertools.combinations(vectors, 2)
    except TypeError as exc:
        wrong_type = type(vectors).__name__
        raise TypeError(f"vectors should be iterable, not {wrong_type}") from exc
    try:
        for vector_1, vector_2 in combinations:
            if np.isclose(vector_1, vector_2).all():
                return np.array(vector_1)
        return None
    except TypeError as exc:
        raise TypeError("elements of vectors are not array_like") from exc
