"""Utility functions relevant to ReferenceStructure."""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    get_type_error,
    verify_positions,
    get_shape_error,
)


def are_collinear(vector_1: NDArray[np.float64], vector_2: NDArray[np.float64]) -> bool:
    """Return whether or not two vectors are collinear.

    Parameters
    ----------
    vector_1
        ndarray with shape (M,)
    vector_2
        ndarray with shape (M,)

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

    Returns
    -------
    int
        first index of non-orthogonal vector, otherwise -1

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

    Returns
    -------
    int
        first index of non-collinear vector, otherwise -1

    """
    # This implementation could be made more efficient.
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
    # This implementation could be made more efficient.
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
    # Ensure no atom is at unit cell boundary by shifting
    # center of mass
    center_of_mass_shift = np.array([0.5, 0.5, 0.5]) - np.mean(
        fractional_positions, axis=0
    )

    permutation_matrices = []
    for rotation, translation in zip(rotations, translations):
        permutation_matrices.append(
            _get_fractional_positions_permutation_matrix(
                displace_fractional_positions(
                    fractional_positions, center_of_mass_shift
                ),
                transform_fractional_positions(
                    fractional_positions, rotation, translation + center_of_mass_shift
                ),
            )
        )
    return np.array(permutation_matrices)


def _get_fractional_positions_permutation_matrix(
    reference_positions: NDArray[np.float64], permuted_positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Calculate a permutation matrix between reference and permuted positions.

    .. warning::
        Arguments must be true permutations of each other. This function does not
        correct for periodic boundary conditions, so it needs to be supplied a
        structure without atoms at the unit cell boundaries.

    Parameters
    ----------
    reference_positions
        A 2D array with shape (N,3)
    permuted_positions
        A 2D array with shape (N,3).

    """
    reference_positions = apply_pbc(reference_positions)
    permuted_positions = apply_pbc(permuted_positions)

    argsort_reference = np.lexsort(
        (
            reference_positions[:, 2],
            reference_positions[:, 1],
            reference_positions[:, 0],
        )
    )
    argsort_permuted = np.lexsort(
        (permuted_positions[:, 2], permuted_positions[:, 1], permuted_positions[:, 0])
    )
    sorted_reference = reference_positions[argsort_reference]
    sorted_permuted = permuted_positions[argsort_permuted]
    if not np.isclose(sorted_reference, sorted_permuted).all():
        raise ValueError("permuted is not a permutation of reference")

    permutation_matrix = np.zeros((len(reference_positions), len(reference_positions)))
    permutation_matrix[tuple(argsort_reference), tuple(argsort_permuted)] = 1

    return permutation_matrix


def transform_fractional_positions(
    positions: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform fractional coordinates under periodic boundary conditions."""
    verify_positions("positions", positions)
    positions = apply_pbc(positions)
    try:
        rotated = positions @ rotation
    except TypeError as exc:
        raise get_type_error("rotation", rotation, "ndarray") from exc
    except ValueError as exc:
        raise get_shape_error("rotation", rotation, "(3,3)") from exc
    rotated = apply_pbc(rotated)
    return displace_fractional_positions(rotated, translation)


def displace_fractional_positions(
    positions: NDArray[np.float64],
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add fractional positions together under periodic boundary conditions."""
    positions = apply_pbc(positions)
    displacement = apply_pbc_displacement(displacement)

    return apply_pbc(positions + displacement)


def calculate_displacement(
    positions_1: NDArray[np.float64],
    positions_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate displacement between two fractional positions.

    Respects periodic boundary conditions.
    """
    positions_1 = apply_pbc(positions_1)
    positions_2 = apply_pbc(positions_2)

    return apply_pbc_displacement(positions_1 - positions_2)


def apply_pbc(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return fractional positions such that all coordinates are b/t 0 and 1."""
    try:
        return positions - positions // 1
    except TypeError as exc:
        raise get_type_error("positions", positions, "ndarray") from exc


def apply_pbc_displacement(displacement: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return fractional displacement such as all coordinates are b/t -0.5 and 0.5."""
    try:
        return np.where(displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1)
    except TypeError as exc:
        raise get_type_error("displacement", displacement, "ndarray") from exc