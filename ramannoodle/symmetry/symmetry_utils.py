"""Utility routines relevant to symmetry."""

import numpy as np
from numpy.typing import NDArray


def are_collinear(vector_1: NDArray[np.float64], vector_2: NDArray[np.float64]) -> bool:
    """Checks if two vectors are collinear"""
    vector_1_copy = vector_1 / np.linalg.norm(vector_1)
    vector_2_copy = vector_2 / np.linalg.norm(vector_2)
    dot_product = vector_1_copy.dot(vector_2_copy)
    result: bool = np.isclose(dot_product, 1).all() or np.isclose(dot_product, -1).all()
    return result


def is_orthogonal_to_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Checks if a vector is orthogonal to all vectors in a list. Returns
    first index of non-orthogonal vector, otherwise returns -1"""

    # This implementation could be made more efficient but readability would
    # be sacrificed .
    vector_1_copy = vector_1 / np.linalg.norm(vector_1)

    for index, vector_2 in enumerate(vectors):
        vector_2_copy = vector_2 / np.linalg.norm(vector_2)
        if not np.isclose(np.dot(vector_1_copy, vector_2_copy) + 1, 1).all():
            return index

    return -1


def is_non_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Checks if a vector is non-collinear to all vectors in a list. Returns
    first index of collinear vector if found, otherwise returns -1"""

    # This implementation could be made more efficient but readability would
    # be sacrificed .
    for index, vector_2 in enumerate(vectors):
        if are_collinear(vector_1.flatten(), vector_2.flatten()):
            return index

    return -1


def compute_permutation_matrices(
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
    fractional_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expresses a series of rotation/translations as permutation matrices."""
    permutation_matrices = []
    for rotation, translation in zip(rotations, translations):
        permutation_matrices.append(
            get_fractional_positions_permutation_matrix(
                fractional_positions,
                transform_fractional_positions(
                    fractional_positions, rotation, translation
                ),
            )
        )
    return np.array(permutation_matrices)


def get_fractional_positions_permutation_matrix(
    reference: NDArray[np.float64], permuted: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gets a permutation matrix for fractional positions"""
    assert (0 <= reference).all() and (reference <= 1.0).all()
    assert (0 <= permuted).all() and (permuted <= 1.0).all()

    # Perhaps not the best implementation, but it'll do for now
    permutation_matrix = np.zeros((len(reference), len(reference)))

    for ref_index, ref_position in enumerate(reference):
        for permuted_index, permuted_position in enumerate(permuted):
            difference = calculate_displacement(permuted_position, ref_position)
            distance = np.sum(difference**2)
            if distance < 0.001:
                permutation_matrix[ref_index][permuted_index] = 1
                break
    assert np.isclose(np.sum(permutation_matrix, axis=1), 1).all()
    return permutation_matrix


def transform_fractional_positions(
    positions: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
) -> NDArray[np.float64]:
    """Transforms fractional coordinates and applies periodic boundary
    conditions."""
    assert (0 <= positions).all() and (positions <= 1.0).all()
    rotated = positions @ rotation
    rotated[rotated < 0.0] += 1
    rotated[rotated > 1.0] -= 1
    return add_fractional_positions(rotated, translation)


def add_fractional_positions(
    positions_1: NDArray[np.float64],
    positions_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Adds fractional positions together and applies periodic boundary
    conditions."""
    assert (0 <= positions_1).all() and (positions_1 <= 1.0).all()
    assert (0 <= positions_2).all() and (positions_2 <= 1.0).all()

    result = positions_1 + positions_2
    result[result < 0.0] += 1
    result[result > 1.0] -= 1
    return result


def calculate_displacement(
    positions_1: NDArray[np.float64],
    positions_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Subtracts fractional positions to form a displacement
    and applies periodic boundary conditions."""
    assert (0 <= positions_1).all() and (positions_1 <= 1.0).all()
    assert (0 <= positions_2).all() and (positions_2 <= 1.0).all()

    difference = positions_1 - positions_2
    difference[difference > 0.5] -= 1.0
    difference[difference < -0.5] += 1.0
    return difference
