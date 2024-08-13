"""Utility functions for structures."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    get_type_error,
    verify_positions,
    get_shape_error,
)


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


def displace_positions(
    positions: NDArray[np.float64],
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add fractional positions together under periodic boundary conditions."""
    positions = apply_pbc(positions)
    displacement = apply_pbc_displacement(displacement)

    return apply_pbc(positions + displacement)


def transform_positions(
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
    return displace_positions(rotated, translation)


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
