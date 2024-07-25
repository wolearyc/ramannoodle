"""Utilities for VASP."""

from pathlib import Path
from typing import TextIO
import numpy as np
from numpy.typing import NDArray

from ...dynamics import Phonons
from ...globals import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from .vasp_utils import (
    _get_atomic_symbol_from_potcar_line,
    _read_atomic_symbols_from_outcar,
    _read_eigenvector_from_outcar,
    _read_cartesian_positions_from_outcar,
    _read_polarizability_from_outcar,
    _read_fractional_positions_from_outcar,
    _read_lattice_from_outcar,
)
from ..io_utils import _skip_file_until_line_contains


def load_phonons_from_outcar(path: Path) -> Phonons:
    """Extracts phonons from an OUTCAR"""
    wavenumbers = []
    eigenvectors = []

    with open(path, "r", encoding="utf-8") as outcar_file:

        # get atom information
        atomic_symbols = _read_atomic_symbols_from_outcar(outcar_file)
        atomic_weights = np.array([ATOMIC_WEIGHTS[symbol] for symbol in atomic_symbols])
        num_atoms = len(atomic_symbols)
        degrees_of_freedom = num_atoms * 3

        # read in eigenvectors/eigenvalues
        _ = _skip_file_until_line_contains(
            outcar_file, "Eigenvectors and eigenvalues of the dynamical matrix"
        )
        for _ in range(degrees_of_freedom):
            line = _skip_file_until_line_contains(outcar_file, "cm-1")
            if "f/i" in line:  # if complex
                wavenumbers.append(-float(line.split()[6]))  # set negative wavenumber
            else:
                wavenumbers.append(float(line.split()[7]))
            eigenvectors.append(_read_eigenvector_from_outcar(outcar_file, num_atoms))

        # Divide eigenvectors by sqrt(mass) to get displacements
        wavenumbers = np.array(wavenumbers)
        eigenvectors = np.array(eigenvectors)
        displacements = eigenvectors / np.sqrt(atomic_weights)[:, np.newaxis]

        assert len(wavenumbers) == len(displacements)
        assert len(wavenumbers) == degrees_of_freedom

        return Phonons(wavenumbers, displacements)


def load_positions_and_polarizability_from_outcar(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extracts the atom positions and polarizability tensor from an OUTCAR"""

    with open(path, "r", encoding="utf-8") as outcar_file:
        num_atoms = len(_read_atomic_symbols_from_outcar(outcar_file))
        positions = _read_cartesian_positions_from_outcar(outcar_file, num_atoms)
        polarizability = _read_polarizability_from_outcar(outcar_file)
        return positions, polarizability


def load_symmetry_cell_from_outcar(
    path: Path,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Extracts a symmetry information (as tuple) from an OUTCAR."""

    lattice = np.array([])
    fractional_positions = np.array([])
    atomic_numbers = np.array([], dtype=np.int32)

    with open(path, "r", encoding="utf-8") as outcar_file:
        atomic_symbols = _read_atomic_symbols_from_outcar(outcar_file)
        atomic_numbers = np.array([ATOMIC_NUMBERS[symbol] for symbol in atomic_symbols])
        lattice = _read_lattice_from_outcar(outcar_file)
        fractional_positions = _read_fractional_positions_from_outcar(
            outcar_file, len(atomic_symbols)
        )

    return (atomic_numbers, lattice, fractional_positions)
