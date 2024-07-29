"""Utilities for VASP.

One should be careful about running these functions, as they will often only partially
read through a given file. Each function must be run in a specific order.

"""

from typing import TextIO
import numpy as np
from numpy.typing import NDArray

from ..io_utils import _skip_file_until_line_contains


def _get_atomic_symbol_from_potcar_line(line: str) -> str:
    """Extract atomic symbol from a POTCAR line in a VASP OUTCAR file."""
    return line.split()[-2].split("_")[0]


def _read_atomic_symbols_from_outcar(outcar_file: TextIO) -> list[str]:
    """Read atomic symbols from a VASP OUTCAR file."""
    potcar_symbols: list[str] = []
    line = _skip_file_until_line_contains(outcar_file, "POTCAR:    ")
    potcar_symbols.append(_get_atomic_symbol_from_potcar_line(line))
    for line in outcar_file:
        if "POTCAR" not in line:
            break
        potcar_symbols.append(_get_atomic_symbol_from_potcar_line(line))

    # HACK: We read the next line and clean up as appropriate.
    # I wish the OUTCAR format was easier to parse, but alas, here we are.
    line = outcar_file.readline()
    if "VRHFIN" in line:
        potcar_symbols.pop()  # We read one too many!

    # Get atom numbers
    line = _skip_file_until_line_contains(outcar_file, "ions per type")
    atomic_numbers = [int(item) for item in line.split()[4:]]

    atomic_symbols = []
    for symbol, number in zip(potcar_symbols, atomic_numbers):
        atomic_symbols += [symbol] * number
    return atomic_symbols


def _read_eigenvector_from_outcar(
    outcar_file: TextIO, num_atoms: int
) -> NDArray[np.float64]:
    """Read the next available phonon eigenvector from a VASP OUTCAR file."""
    eigenvector: list[list[float]] = []
    for line in outcar_file:
        if len(eigenvector) == num_atoms:
            break
        if "X" in line:
            continue
        eigenvector.append([float(item) for item in line.split()[3:]])
    return np.array(eigenvector)


def _read_cartesian_positions_from_outcar(
    outcar_file: TextIO, num_atoms: int
) -> NDArray[np.float64]:
    """Read atomic cartesian positions from a VASP OUTCAR file."""
    _ = _skip_file_until_line_contains(
        outcar_file, "position of ions in cartesian coordinates  (Angst):"
    )

    cartesian_coordinates = []
    for _ in range(num_atoms):
        line = outcar_file.readline()
        cartesian_coordinates.append([float(item) for item in line.split()])

    return np.array(cartesian_coordinates)


def _read_fractional_positions_from_outcar(
    outcar_file: TextIO, num_atoms: int
) -> NDArray[np.float64]:
    """Read atomic fractional positions from a VASP OUTCAR file."""
    _ = _skip_file_until_line_contains(
        outcar_file, "position of ions in fractional coordinates (direct lattice)"
    )

    fractional_coordinates = []
    for _ in range(num_atoms):
        line = outcar_file.readline()
        fractional_coordinates.append([float(item) for item in line.split()])

    return np.array(fractional_coordinates)


def _read_polarizability_from_outcar(outcar_file: TextIO) -> NDArray[np.float64]:
    """Read polarizability from a VASP OUTCAR file.

    In actuality, we read the macroscopic dielectric tensor including local field
    effects.

    """
    _ = _skip_file_until_line_contains(
        outcar_file,
        "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT",
    )
    outcar_file.readline()

    polarizability = []
    for _ in range(3):
        line = outcar_file.readline()
        polarizability.append([float(item) for item in line.split()])
    return np.array(polarizability)


def _get_lattice_vector_from_outcar_line(line: str) -> NDArray[np.float64]:
    """Extract lattice vector from an appropriate line in a VASP OUTCAR file."""
    return np.array([float(item) for item in line.split()[0:3]])


def _read_lattice_from_outcar(outcar_file: TextIO) -> NDArray[np.float64]:
    """Read all three lattice vectors (in angstroms) from a VASP OUTCAR file."""
    # HACK: if symmetry is turned on, outcar will write additional
    # lattice vectors. We must skip ahead.
    _ = _skip_file_until_line_contains(
        outcar_file,
        "Write flags",
    )
    _ = _skip_file_until_line_contains(
        outcar_file,
        "direct lattice vectors      ",
    )

    lattice = []
    for _ in range(3):
        line = outcar_file.readline()
        lattice.append(_get_lattice_vector_from_outcar_line(line))

    return np.array(lattice)
