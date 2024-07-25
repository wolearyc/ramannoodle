"""Testing for the polarizability."""

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import (
    are_collinear,
    check_orthogonal,
)


@pytest.mark.parametrize(
    "vector_1, vector_2, known",
    [
        (np.array([-5.0, -5.0, 1.0]), np.array([1.0, 1.0, 0.0]), False),
        (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), False),
        (np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, -3.0]), True),
        (np.array([0.0, 0.0, -1.0]), np.array([1.0, 3.0, 1.0]), False),
    ],
)
def test_are_collinear(
    vector_1: NDArray[np.float64], vector_2: NDArray[np.float64], known: bool
) -> None:
    """Test"""
    assert are_collinear(vector_1, vector_2) == known


@pytest.mark.parametrize(
    "vector_1, vectors, known",
    [
        (np.array([1, 0, 0]), np.array([[0, 1, 0], [0, 0, 1], [-1, -1, 0]]), 2),
        (np.array([1, 1, 0]), np.array([[0, 0, 1], [-1, 1, 0]]), -1),
    ],
)
def test_check_orthogonal(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]], known: int
) -> None:
    """Test"""
    assert check_orthogonal(vector_1, vectors) == known
