"""Routines for interacting with files used and produced by DFT codes."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from ..dynamics import Phonons
from ..symmetry import StructuralSymmetry
from . import vasp

# These dictionaries map between file_format's and appropriate loading functions.
_PHONON_LOADERS = {"outcar": vasp.read_phonons_from_outcar}
_POSITION_AND_POLARIZABILITY_LOADERS = {
    "outcar": vasp.read_positions_and_polarizability_from_outcar
}
_STRUCTURAL_SYMMETRY_LOADERS = {"outcar": vasp.read_structural_symmetry_from_outcar}


def read_phonons(filepath: str | Path, file_format: str) -> Phonons:
    """Read phonons from a file.

    Parameters
    ----------
    filepath
    file_format

    Returns
    -------
    :

    Raises
    ------
    InvalidFileException
    ValueError
    """
    try:
        return _PHONON_LOADERS[file_format](filepath)
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

    Returns
    -------
    :
        2-tuple, whose first element is the fractional positions, a 2D array with shape
        (N,3). The second element is the polarizability, a 2D array with shape (3,3).

    Raises
    ------
    InvalidFileException
    ValueError
    """
    try:
        return _POSITION_AND_POLARIZABILITY_LOADERS[file_format](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_structural_symmetry(
    filepath: str | Path, file_format: str
) -> StructuralSymmetry:
    """Read structural symmetry from a file.

    Parameters
    ----------
    filepath
    file_format

    Returns
    -------
    StructuralSymmetry

    Raises
    ------
    InvalidFileException
        If the OUTCAR has an unexpected format.
    SymmetryException
        If OUTCAR was read but the symmetry search failed
    ValueError
    """
    try:
        return _STRUCTURAL_SYMMETRY_LOADERS[file_format](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc
