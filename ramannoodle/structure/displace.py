"""Routines for generating and writing displaced structure."""

# Design note:
# These routines are not implemented in ReferenceStructure to give
# greater modularity. For example, when different displacement methods are added (such
# as Monte Carlo rattling or random displacements), we'd rather not add to code to
# Reference Structure. These functions stand alone, just like the IO functions.

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.structure.structure_utils import displace_positions
from ramannoodle.exceptions import (
    get_type_error,
    get_shape_error,
    verify_ndarray_shape,
    verify_list_len,
)
from ramannoodle.io.io_utils import pathify_as_list
import ramannoodle.io.generic as rn_io


def get_displaced_positions(
    ref_structure: ReferenceStructure,
    cart_displacement: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Return list of displaced positions given a displacement and amplitudes.

    Parameters
    ----------
    ref_structure
        reference structure of N atoms
    cart_displacement
        2D array with shape (N,3)
    amplitudes
        1D array with shape (M,)

    Returns
    -------
    :
        1D list of length M
    """
    try:
        cart_displacement = cart_displacement / float(np.linalg.norm(cart_displacement))
    except TypeError as err:
        raise get_type_error("cart_displacement", cart_displacement, "ndarray") from err
    verify_ndarray_shape("amplitudes", amplitudes, (None,))

    positions = []
    for amplitude in amplitudes:
        try:
            displacement = ref_structure.get_frac_displacement(
                cart_displacement * amplitude
            )
        except ValueError as err:
            raise get_shape_error(
                "cart_displacement",
                cart_displacement,
                f"({len(ref_structure.positions)}, 3)",
            ) from err
        positions.append(displace_positions(ref_structure.positions, displacement))

    return positions


def write_displaced_structures(  # pylint: disable=too-many-arguments
    ref_structure: ReferenceStructure,
    cart_displacement: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    file_paths: str | Path | list[str] | list[Path],
    file_format: str,
    overwrite: bool = False,
) -> None:
    """Write displaced structures to files.

    Parameters
    ----------
    ref_structure
        reference structure of N atoms
    cart_displacement
        2D array with shape (N,3)
    amplitudes
        1D array with shape (M,)
    file_paths
    file_format
        supports: "poscar"
    overwrite
    """
    file_paths = pathify_as_list(file_paths)
    position_list = get_displaced_positions(
        ref_structure, cart_displacement, amplitudes
    )
    verify_list_len("file_paths", file_paths, len(position_list))

    for position, filepath in zip(position_list, file_paths):
        rn_io.write_structure(
            ref_structure.lattice,
            ref_structure.atomic_numbers,
            position,
            filepath,
            file_format,
            overwrite,
        )


def get_ast_displaced_positions(
    ref_structure: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Return list of displaced positions with an atom displaced along a direction.

    Parameters
    ----------
    ref_structure
        reference structure of N atoms
    atom_index
    cart_direction
        1D array with shape (3,)
    amplitudes
        1D array with shape (M,)

    Returns
    -------
    :
        1D list of length M
    """
    try:
        cart_direction = cart_direction / float(np.linalg.norm(cart_direction))
    except TypeError as err:
        raise get_type_error("cart_direction", cart_direction, "ndarray") from err
    positions = []
    for amplitude in amplitudes:
        cart_displacement = ref_structure.positions * 0
        try:
            cart_displacement[atom_index] = cart_direction * amplitude
        except IndexError as err:
            raise IndexError(f"invalid atom_index: {atom_index}") from err
        displacement = ref_structure.get_frac_displacement(cart_displacement)
        positions.append(displace_positions(ref_structure.positions, displacement))

    return positions


def write_ast_displaced_structures(  # pylint: disable=too-many-arguments
    ref_structure: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    file_paths: str | Path | list[str] | list[Path],
    file_format: str,
    overwrite: bool = False,
) -> None:
    """Return displaced structures with an atom displaced along a direction.

    Parameters
    ----------
    ref_structure
        reference structure of N atoms
    atom_index
    cart_direction
        1D array with shape (3,)
    amplitudes
        1D array with shape (M,)
    file_paths
    file_format
        supports: "poscar"
    overwrite

    Returns
    -------
    :
        1D list of length M
    """
    file_paths = pathify_as_list(file_paths)
    position_list = get_ast_displaced_positions(
        ref_structure, atom_index, cart_direction, amplitudes
    )
    verify_list_len("file_paths", file_paths, len(position_list))

    for position, filepath in zip(position_list, file_paths):
        rn_io.write_structure(
            ref_structure.lattice,
            ref_structure.atomic_numbers,
            position,
            filepath,
            file_format,
            overwrite,
        )
