"""Utility functions relevant to symmetry."""

import numpy as np
from numpy.typing import NDArray


def are_collinear(vector_1: NDArray[np.float64], vector_2: NDArray[np.float64]) -> bool:
    """Return whether or not two vectors are collinear."""
    vector_1_copy = vector_1 / np.linalg.norm(vector_1)
    vector_2_copy = vector_2 / np.linalg.norm(vector_2)
    dot_product = vector_1_copy.dot(vector_2_copy)
    result: bool = np.isclose(dot_product, 1).all() or np.isclose(dot_product, -1).all()
    return result


def is_orthogonal_to_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Check whether a given vector is orthogonal to a list of others.

    Returns
    -------
    int
        first index of non-orthogonal vector, otherwise -1

    """
    # This implementation could be made more efficient but readability would
    # be sacrificed .
    vector_1_copy = vector_1 / np.linalg.norm(vector_1)

    for index, vector_2 in enumerate(vectors):
        vector_2_copy = vector_2 / np.linalg.norm(vector_2)
        if not np.isclose(
            np.dot(vector_1_copy.flatten(), vector_2_copy.flatten()) + 1, 1
        ).all():
            return index

    return -1


def is_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Check if a given vector is collinear to a list of others.

    Returns
    -------
    int
        first index of non-collinear vector, otherwise -1

    """
    # This implementation could be made more efficient but readability would
    # be sacrificed.
    for index, vector_2 in enumerate(vectors):
        if not are_collinear(vector_1.flatten(), vector_2.flatten()):
            return index

    return -1


def is_non_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]]
) -> int:
    """Check if a given vector is non-collinear to a list of others.

    Returns
    -------
    int
        first index of collinear vector, otherwise -1

    """
    # This implementation could be made more efficient but readability would
    # be sacrificed.
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
    """Calculate a permutation matrix given permuted fractional positions."""
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
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform fractional coordinates under periodic boundary conditions."""
    assert (0 <= positions).all() and (positions <= 1.0).all()
    rotated = positions @ rotation
    rotated[rotated < 0.0] += 1
    rotated[rotated > 1.0] -= 1
    return displace_fractional_positions(rotated, translation)


def displace_fractional_positions(
    positions: NDArray[np.float64],
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add fractional positions together under periodic boundary conditions."""
    assert (0 <= positions).all() and (positions <= 1.0).all()

    result = positions + displacement
    result[result < 0.0] += 1
    result[result > 1.0] -= 1
    return result


def calculate_displacement(
    positions_1: NDArray[np.float64],
    positions_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Subtracts fractional positions under periodic boundary conditions.

    Returns a displacement.

    """
    assert (0 <= positions_1).all() and (positions_1 <= 1.0).all()
    assert (0 <= positions_2).all() and (positions_2 <= 1.0).all()

    difference = positions_1 - positions_2
    difference[difference > 0.5] -= 1.0
    difference[difference < -0.5] += 1.0
    return difference
