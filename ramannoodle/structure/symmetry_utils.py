"""Utility functions relevant to symmetry."""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    get_type_error,
)


def are_collinear(vector_1: NDArray[np.float64], vector_2: NDArray[np.float64]) -> bool:
    """Return whether or not two vectors are collinear.

    Parameters
    ----------
    vector_1
        | 1D array with shape (M,).
    vector_2
        | 1D array with shape (M,).

    """
    try:
        vector_1 = vector_1 / float(np.linalg.norm(vector_1))
    except TypeError as exc:
        raise get_type_error("vector_1", vector_1, "ndarray") from exc
    try:
        vector_2 = vector_2 / float(np.linalg.norm(vector_2))
    except TypeError as exc:
        raise get_type_error("vector_2", vector_2, "ndarray") from exc
    try:
        dot_product = vector_1.dot(vector_2)
    except ValueError as exc:
        length_expr = f"{len(vector_1)} != {len(vector_2)}"
        raise ValueError(
            f"vector_1 and vector_2 have different lengths: {length_expr}"
        ) from exc
    return bool(np.isclose(dot_product, 1).all() or np.isclose(dot_product, -1).all())


def is_orthogonal_to_all(
    vector_1: NDArray[np.float64], vectors: Iterable[NDArray[np.float64]]
) -> int:
    """Check whether a given vector is orthogonal to a list of others.

    Parameters
    ----------
    vector_1
        | 1D array with shape (M,).
    vectors
        | Iterable containing 1D arrays with shape (M,).

    Returns
    -------
    :
        First index of non-orthogonal vector, otherwise -1.

    """
    # This implementation could be made more efficient.
    try:
        vector_1 = vector_1 / float(np.linalg.norm(vector_1))
    except TypeError as exc:
        raise get_type_error("vector_1", vector_1, "ndarray") from exc

    for index, vector_2 in enumerate(vectors):
        try:
            vector_2 = vector_2 / np.linalg.norm(vector_2)
        except TypeError as exc:
            raise get_type_error(f"vectors[{index}]", vector_2, "ndarray") from exc

        if not np.isclose(np.dot(vector_1.flatten(), vector_2.flatten()) + 1, 1).all():
            return index

    return -1


def is_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: Iterable[NDArray[np.float64]]
) -> int:
    """Check if a given vector is collinear to a list of others.

    Parameters
    ----------
    vector_1
        | 1D array with shape (M,).
    vectors
        | Iterable containing 1D arrays with shape (M,).

    Returns
    -------
    :
        | First index of non-collinear vector, otherwise -1.

    """
    # This implementation could be made more efficient.
    for index, vector_2 in enumerate(vectors):
        if not are_collinear(vector_1.flatten(), vector_2.flatten()):
            return index

    return -1


def is_non_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: Iterable[NDArray[np.float64]]
) -> int:
    """Check if a given vector is non-collinear to a list of others.

    Parameters
    ----------
    vector_1
        | 1D array with shape (M,).
    vectors
        | Iterable containing 1D arrays with shape (M,).

    Returns
    -------
    :
        First index of collinear vector, otherwise -1.

    """
    # This implementation could be made more efficient.
    for index, vector_2 in enumerate(vectors):
        if are_collinear(vector_1.flatten(), vector_2.flatten()):
            return index

    return -1
