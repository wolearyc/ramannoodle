"""Routines for interacting with files used and produced by DFT codes."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.phonon import Phonons
from ramannoodle.symmetry.structural import ReferenceStructure
from ramannoodle.io import vasp

# These dictionaries map between file_format's and appropriate loading functions.
_PHONON_READERS = {"outcar": vasp.outcar.read_phonons}
_POSITION_AND_POLARIZABILITY_READERS = {
    "outcar": vasp.outcar.read_positions_and_polarizability
}
_REFERENCE_STRUCTURE_READERS = {
    "outcar": vasp.outcar.read_ref_structure,
    "poscar": vasp.poscar.read_ref_structure,
}


def read_phonons(filepath: str | Path, file_format: str) -> Phonons:
    """Read phonons from a file.

    Parameters
    ----------
    filepath
    file_format
        supports: "outcar"

    Returns
    -------
    :

    Raises
    ------
    InvalidFileException
        File has unexpected format.
    FileNotFoundError
        File could not be found.
    """
    try:
        return _PHONON_READERS[file_format](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_positions_and_polarizability(
    filepath: str | Path,
    file_format: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read fractional positions and polarizability from a file.

    Parameters
    ----------
    filepath
    file_format
        supports: "outcar"

    Returns
    -------
    :
        2-tuple, whose first element is the fractional positions, a 2D array with shape
        (N,3). The second element is the polarizability, a 2D array with shape (3,3).

    Raises
    ------
    InvalidFileException
        File has unexpected format.
    FileNotFoundError
        File could not be found.
    """
    try:
        return _POSITION_AND_POLARIZABILITY_READERS[file_format](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_ref_structure(filepath: str | Path, file_format: str) -> ReferenceStructure:
    """Read reference structure from a file.

    Parameters
    ----------
    filepath
    file_format
        supports: "outcar"

    Returns
    -------
    StructuralSymmetry

    Raises
    ------
    InvalidFileException
        File has unexpected format.
    FileNotFoundError
        File could not be found.
    """
    try:
        return _REFERENCE_STRUCTURE_READERS[file_format](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


_STRUCTURE_WRITERS = {"poscar": vasp.poscar.write_structure}


def write_structure(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    fractional_positions: NDArray[np.float64],
    filepath: str | Path,
    file_format: str,
    **options: str,
) -> None:
    """Write structure to file.

    Parameters
    ----------
    lattice
    atomic_numbers
    fractional_positions
    filepath
    file_format
        supports: "poscar"
    overwrite
    options
        Additional, file-format-dependent options. See IO.

    Raises
    ------
    InvalidFileException
        File has unexpected format.
    """
    try:
        _STRUCTURE_WRITERS[file_format](
            lattice,
            atomic_numbers,
            fractional_positions,
            filepath,
            **options,
        )
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc
