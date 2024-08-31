"""Functions for generating and writing displaced structures.

These functions are useful for preparing polarizability calculations needed for
:class:`~.InterpolationModel` and :class:`~.ARTModel`.

"""

# Design note:
# These functions are not implemented in ReferenceStructure to give
# greater modularity. For example, when different displacement methods are added (such
# as Monte Carlo rattling or random displacements), we'd rather not add code to
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
import ramannoodle.io.generic as generic_io


def get_displaced_positions(
    ref_structure: ReferenceStructure,
    cart_displacement: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Return positions displaced along a certain displacement.

    Parameters
    ----------
    ref_structure
        | Reference structure containing N atoms.
    cart_displacement
        (Å) 2D array with shape (N,3).

        Magnitude is arbitrary.
    amplitudes
        | (Å) 1D array with shape (M,).

    Returns
    -------
    :
        (fractional) List of length M containing 2D arrays with shape (N,3).

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
    filepaths: str | Path | list[str] | list[Path],
    file_format: str,
    overwrite: bool = False,
) -> None:
    """Write displaced structures to files.

    Parameters
    ----------
    ref_structure
        | Reference structure containing N atoms
    cart_displacement
        (Å) 2D array with shape (N,3).

        Magnitude is arbitrary.
    amplitudes
        | (Å) 1D array with shape (M,).
    filepaths
    file_format
        | Supports ``"poscar"`` (see :ref:`Supported formats`).
    overwrite
        | Overwrite the file if it exists.
    """
    filepaths = pathify_as_list(filepaths)
    position_list = get_displaced_positions(
        ref_structure, cart_displacement, amplitudes
    )
    verify_list_len("filepaths", filepaths, len(position_list))

    for position, filepath in zip(position_list, filepaths):
        generic_io.write_structure(
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
    """Return displaced positions with a single atom displaced along a direction.

    Parameters
    ----------
    ref_structure
        | Reference structure containing N atoms.
    atom_index
    cart_direction
        (Å) 1D array with shape (3,).

        Magnitude is arbitrary.
    amplitudes
        | (Å) 1D array with shape (M,).

    Returns
    -------
    :
        (fractional) List of length M containing 2D arrays with shape (N,3).
    """
    try:
        cart_direction = cart_direction / float(np.linalg.norm(cart_direction))
    except TypeError as err:
        raise get_type_error("cart_direction", cart_direction, "ndarray") from err
    cart_displacement = ref_structure.positions * 0
    try:
        cart_displacement[atom_index] = cart_direction
    except ValueError as err:
        raise get_shape_error("cart_direction", cart_direction, "(3,)") from err
    except IndexError as err:
        raise IndexError(f"invalid atom_index: {atom_index}") from err

    return get_displaced_positions(ref_structure, cart_displacement, amplitudes)


def write_ast_displaced_structures(  # pylint: disable=too-many-arguments
    ref_structure: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    filepaths: str | Path | list[str] | list[Path],
    file_format: str,
    overwrite: bool = False,
) -> None:
    """Write displaced structures with a single atom displaced along a direction.

    Parameters
    ----------
    ref_structure
        Reference structure containing N atoms.
    atom_index
    cart_direction
        | (Å) 1D array with shape (3,).

        Magnitude is arbitrary.
    amplitudes
        | (Å) 1D array with shape (M,).
    filepaths
    file_format
        | Supports ``"poscar"`` (see :ref:`Supported formats`).
    overwrite
        | Overwrite the file if it exists.
    """
    filepaths = pathify_as_list(filepaths)
    position_list = get_ast_displaced_positions(
        ref_structure, atom_index, cart_direction, amplitudes
    )
    verify_list_len("filepaths", filepaths, len(position_list))

    for position, filepath in zip(position_list, filepaths):
        generic_io.write_structure(
            ref_structure.lattice,
            ref_structure.atomic_numbers,
            position,
            filepath,
            file_format,
            overwrite,
        )
