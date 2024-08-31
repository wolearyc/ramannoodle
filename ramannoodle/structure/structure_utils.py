"""Utility functions for structures."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    get_type_error,
    verify_positions,
    get_shape_error,
)


def apply_pbc(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return fractional positions such that all coordinates are between 0 and 1.

    Parameters
    ----------
    positions
        | (fractional) 2D array with shape (N,3) where N is the number of atoms.

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3).
    """
    try:
        return positions - positions // 1
    except TypeError as exc:
        raise get_type_error("positions", positions, "ndarray") from exc


def apply_pbc_displacement(displacement: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return fractional displacement such as all coordinates are between -0.5 and 0.5.

    Parameters
    ----------
    displacement
        | (fractional) 2D array with shape (N,3) where N is the number of atoms.

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3).
    """
    try:
        return np.where(displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1)
    except TypeError as exc:
        raise get_type_error("displacement", displacement, "ndarray") from exc


def displace_positions(
    positions: NDArray[np.float64],
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Displace positions, respecting periodic boundary conditions.

    Parameters
    ----------
    positions
        | (fractional) 2D array with shape (N,3) where N is the number of atoms.
    displacement
        | (fractional) 2D array with shape (N,3).

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3).
    """
    positions = apply_pbc(positions)
    displacement = apply_pbc_displacement(displacement)

    return apply_pbc(positions + displacement)


def transform_positions(
    positions: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform positions, respecting periodic boundary conditions.

    Parameters
    ----------
    positions
        | (fractional) 2D array with shape (N,3) where N is the number of atoms
    rotation
        | 2D array with shape (3,3).
    translation
        | (fractional) 1D array with shape (3,).

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3).
    """
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


def calc_displacement(
    positions_1: NDArray[np.float64],
    positions_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate minimum displacement between two fractional positions.

    Respects periodic boundary conditions.

    Parameters
    ----------
    positions_1
        | (fractional) 2D array with shape (N,3) where N is the number of atoms.
    positions_2
        | (fractional) 2D array with shape (N,3).

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3).

        Displacement is from ``positions_1`` to ``positions_2``.
    """
    positions_1 = apply_pbc(positions_1)
    positions_2 = apply_pbc(positions_2)

    return apply_pbc_displacement(positions_2 - positions_1)
