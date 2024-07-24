"""Utilities for VASP."""

from pathlib import Path
from typing import TextIO
import numpy as np
from numpy.typing import NDArray

from ...dynamics import Phonons
from ...globals import ATOMIC_MASSES
from .vasp_utils import (
    _get_atom_symbol_from_potcar_line,
    _read_atom_symbols_from_outcar,
    _read_eigenvector_from_outcar,
)
from ..io_utils import _skip_file_until_line_contains


def load_phonons_from_outcar(path: Path) -> Phonons:
    """Extracts phonons from an OUTCAR"""
    wavenumbers = []
    eigenvectors = []

    with open(path, "r", encoding="utf-8") as outcar_file:

        # get atom information
        atom_symbols = _read_atom_symbols_from_outcar(outcar_file)
        atom_masses = np.array([ATOMIC_MASSES[symbol] for symbol in atom_symbols])
        num_atoms = len(atom_symbols)
        degrees_of_freedom = num_atoms * 3 - 3

        # read in eigenvectors/eigenvalues
        _ = _skip_file_until_line_contains(
            outcar_file, "Eigenvectors and eigenvalues of the dynamical matrix"
        )
        for _ in range(degrees_of_freedom):
            line = _skip_file_until_line_contains(outcar_file, "cm-1")
            wavenumbers.append(float(line.split()[7]))
            eigenvectors.append(_read_eigenvector_from_outcar(outcar_file, num_atoms))

        # Divide eigenvectors by sqrt(mass) to get displacements
        wavenumbers = np.array(wavenumbers)
        eigenvectors = np.array(eigenvectors)
        displacements = eigenvectors / np.sqrt(atom_masses)[:, np.newaxis]

        assert len(wavenumbers) == len(displacements)
        assert len(wavenumbers) == degrees_of_freedom

    return Phonons(wavenumbers, eigenvectors)
