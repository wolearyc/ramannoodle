"""Utilities for VASP."""

from pathlib import Path
from typing import TextIO
import numpy as np
from numpy.typing import NDArray

from ..dynamics import Phonons
from ..exceptions import NoMatchingLineFoundException
from ..globals import ATOMIC_MASSES


def _skip_file_until_line_contains(file: TextIO, content: str) -> str:
    """Reads through a file until a line containing content is found."""
    for line in file:
        if content in line:
            return line
    raise NoMatchingLineFoundException(content)


def _get_atom_symbol_from_potcar_line(potcar_line: str) -> str:
    """e.g. "POTCAR:    PAW_PBE Ti_pv 07Sep2000" -> "Ti" """
    return potcar_line.split()[-2].split("_")[0]


def _read_atom_symbols_from_outcar(outcar_file: TextIO) -> list[str]:
    """Reads through outcar and returns the atom symbol list."""
    potcar_symbols: list[str] = []
    line = _skip_file_until_line_contains(outcar_file, "POTCAR:    ")
    potcar_symbols.append(_get_atom_symbol_from_potcar_line(line))
    for line in outcar_file:
        if "POTCAR" not in line:
            break
        potcar_symbols.append(_get_atom_symbol_from_potcar_line(line))

    # HACK: We read the next line and clean up as appropriate.
    # I wish the OUTCAR format was better, but alas, here we are.
    line = outcar_file.readline()
    if "VRHFIN" in line:
        potcar_symbols.pop()  # We read one too many!

    # Get atom numbers
    line = _skip_file_until_line_contains(outcar_file, "ions per type")
    atom_numbers = [int(item) for item in line.split()[4:]]

    atom_symbols = []
    for symbol, number in zip(potcar_symbols, atom_numbers):
        atom_symbols += [symbol] * number
    return atom_symbols


def _read_eigenvector_from_outcar(
    outcar_file: TextIO, num_atoms: int
) -> NDArray[np.float64]:
    """Reads a single eigenvector from an OUTCAR."""
    eigenvector: list[list[float]] = []
    for line in outcar_file:
        if len(eigenvector) == num_atoms:
            break
        if "X" in line:
            continue
        eigenvector.append([float(item) for item in line.split()[3:]])
    return np.array(eigenvector)


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
