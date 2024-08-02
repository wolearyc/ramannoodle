"""Functions for interacting with VASP input and output files."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from ...dynamics import Phonons
from ...symmetry import StructuralSymmetry
from ...globals import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from .vasp_utils import (
    _read_atomic_symbols_from_outcar,
    _read_eigenvector_from_outcar,
    _read_polarizability_from_outcar,
    _read_fractional_positions_from_outcar,
    _read_lattice_from_outcar,
)
from ..io_utils import _skip_file_until_line_contains, pathify
from ...exceptions import NoMatchingLineFoundException, InvalidFileException


def load_phonons_from_outcar(filepath: str | Path) -> Phonons:
    """Extract phonons from a VASP OUTCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    :

    Raises
    ------
    InvalidFileException
        If the OUTCAR has an unexpected format.
    """
    wavenumbers = []
    eigenvectors = []

    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as outcar_file:

        # get atom information
        atomic_symbols = _read_atomic_symbols_from_outcar(outcar_file)
        atomic_weights = np.array([ATOMIC_WEIGHTS[symbol] for symbol in atomic_symbols])
        num_atoms = len(atomic_symbols)
        num_degrees_of_freedom = num_atoms * 3

        # read in eigenvectors/eigenvalues
        try:
            _ = _skip_file_until_line_contains(
                outcar_file, "Eigenvectors and eigenvalues of the dynamical matrix"
            )
        except NoMatchingLineFoundException as exc:
            raise InvalidFileException(
                "eigenvector/eigenvalues block not found"
            ) from exc
        for _ in range(num_degrees_of_freedom):
            try:
                line = _skip_file_until_line_contains(outcar_file, "cm-1")
                if "f/i" in line:  # if complex
                    wavenumbers.append(
                        -float(line.split()[6])  # set negative wavenumber
                    )
                else:
                    wavenumbers.append(float(line.split()[7]))
            except (NoMatchingLineFoundException, TypeError, IndexError) as exc:
                raise InvalidFileException("eigenvalue could not be parsed") from exc
            eigenvectors.append(_read_eigenvector_from_outcar(outcar_file, num_atoms))

        # Divide eigenvectors by sqrt(mass) to get cartesian displacements
        wavenumbers = np.array(wavenumbers)
        eigenvectors = np.array(eigenvectors)
        cartesian_displacements = eigenvectors / np.sqrt(atomic_weights)[:, np.newaxis]

        return Phonons(wavenumbers, cartesian_displacements)


def load_positions_and_polarizability_from_outcar(
    filepath: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract fractional positions and polarizability from a VASP OUTCAR file.

    The polarizability returned by VASP is, in fact, a dielectric tensor. However,
    this is inconsequential to the calculation of Raman spectra.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        2-tuple, whose first element is the fractional positions, a 2D array with shape
        (N,3). The second element is the polarizability, a 2D array with shape (3,3).

    Raises
    ------
    InvalidFileException
        If the OUTCAR has an unexpected format.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as outcar_file:
        num_atoms = len(_read_atomic_symbols_from_outcar(outcar_file))
        positions = _read_fractional_positions_from_outcar(outcar_file, num_atoms)
        polarizability = _read_polarizability_from_outcar(outcar_file)
        return positions, polarizability


def load_structural_symmetry_from_outcar(
    filepath: str | Path,
) -> StructuralSymmetry:
    """Extract structural symmetry from a VASP OUTCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    StructuralSymmetry

    Raises
    ------
    InvalidFileException
        If the OUTCAR has an unexpected format.
    SymmetryException
        If OUTCAR was read but the symmetry search failed
    """
    lattice = np.array([])
    fractional_positions = np.array([])
    atomic_numbers = np.array([], dtype=np.int32)

    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as outcar_file:
        atomic_symbols = _read_atomic_symbols_from_outcar(outcar_file)
        atomic_numbers = np.array([ATOMIC_NUMBERS[symbol] for symbol in atomic_symbols])
        lattice = _read_lattice_from_outcar(outcar_file)
        fractional_positions = _read_fractional_positions_from_outcar(
            outcar_file, len(atomic_symbols)
        )

    return StructuralSymmetry(atomic_numbers, lattice, fractional_positions)
