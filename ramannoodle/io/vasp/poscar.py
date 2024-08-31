"""Functions for interacting with VASP POSCAR files."""

from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from ramannoodle.io.io_utils import verify_structure, pathify
from ramannoodle.exceptions import InvalidFileException
from ramannoodle.globals import ATOM_SYMBOLS, ATOMIC_NUMBERS
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.io.vasp.outcar import _get_lattice_vector_from_outcar_line

# pylint: disable=R0801


def _read_lattice(file: TextIO) -> NDArray[np.float64]:
    """Read all three lattice vectors (in angstroms) from a VASP POSCAR file.

    Raises
    ------
    InvalidFileException
    """
    try:
        file.readline()
        line = file.readline()
        scale_factor = float(line)
    except ValueError as exc:
        raise InvalidFileException(f"scale factor could not be parsed: {line}") from exc

    lattice = []
    for _ in range(3):
        try:
            line = file.readline()
            vector = _get_lattice_vector_from_outcar_line(line)
        except (EOFError, ValueError) as exc:
            raise InvalidFileException(f"lattice could not be parsed: {line}") from exc
        if vector.shape != (3,):
            raise InvalidFileException(f"lattice could not be parsed: {line}")
        lattice.append(vector)

    return np.array(lattice) * scale_factor


def _read_atomic_symbols(file: TextIO) -> list[str]:
    """Read atomic symbols from a VASP POSCAR file.

    Raises
    ------
    InvalidFileException

    """
    # First read symbols
    line = file.readline()
    symbols = line.split()
    if len(symbols) == 0:
        raise InvalidFileException("no atom symbols found")
    for symbol in symbols:
        if symbol not in ATOMIC_NUMBERS:
            raise InvalidFileException(f"unrecognized atom symbol: {symbol}")

    # Then read ion counts
    line = file.readline()
    str_counts = line.split()
    if len(str_counts) != len(symbols):
        spec = f"{len(str_counts)} != {len(symbols)}"
        raise InvalidFileException(f"wrong number of ion counts: {spec}")
    int_counts = []
    for count in str_counts:
        try:
            int_counts.append(int(count))
        except ValueError as err:
            raise InvalidFileException(f"could not parse counts: {line}") from err

    result = []
    for symbol, int_count in zip(symbols, int_counts):
        result += [symbol] * int_count
    return result


def _read_positions(
    file: TextIO, lattice: NDArray[np.float64], num_atoms: int
) -> NDArray[np.float64]:
    """Read atomic symbols from a VASP POSCAR file.

    Raises
    ------
    InvalidFileException

    """
    cart_mode = False
    label = file.readline()
    if len(label.strip()) == 0 or label[0].strip() == "":
        raise InvalidFileException(
            f"missing first character in coordinate format: '{label}'"
        )
    if label[0].lower() == "s":  # selective dynamics
        label = file.readline()
    if label[0].lower() == "c":
        cart_mode = True
    elif label[0].lower() != "d":
        raise InvalidFileException(f"unrecognized coordinate format: {label}")

    positions = []
    for _ in range(num_atoms):
        try:
            line = file.readline()
            position = [float(item) for item in line.split()[0:3]]
        except (EOFError, ValueError, IndexError) as exc:
            raise InvalidFileException(
                f"positions could not be parsed: {line}"
            ) from exc
        if len(position) != 3:
            raise InvalidFileException(f"positions could not be parsed: {line}")
        positions.append(position)

    if cart_mode:
        return np.array(positions) @ np.linalg.inv(lattice)
    return np.array(positions)


def read_positions(
    filepath: str | Path,
) -> NDArray[np.float64]:
    """Read fractional positions from a VASP POSCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3) where N is the number of atoms.

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        lattice = _read_lattice(file)
        atomic_symbols = _read_atomic_symbols(file)
        positions = _read_positions(file, lattice, len(atomic_symbols))
        return positions


def read_ref_structure(
    filepath: str | Path,
) -> ReferenceStructure:
    """Read reference structure from a VASP POSCAR file.

    Parameters
    ----------
    filepath

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    SymmetryException
        Structural symmetry determination failed.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        lattice = _read_lattice(file)
        atomic_symbols = _read_atomic_symbols(file)
        atomic_numbers = [ATOMIC_NUMBERS[symbol] for symbol in atomic_symbols]
        positions = _read_positions(file, lattice, len(atomic_symbols))
        return ReferenceStructure(atomic_numbers, lattice, positions)


def _get_symbols_str(atomic_numbers: list[int]) -> str:
    """Return 2-line string giving elements and counts."""
    unique_numbers = []
    current_number = atomic_numbers[0]
    for number in atomic_numbers:
        if number != current_number and number in unique_numbers:
            raise ValueError(f"atomic number not grouped: {number}")
        if number != current_number:
            unique_numbers.append(current_number)
            current_number = number
    unique_numbers.append(current_number)

    result = "   " + "  ".join([ATOM_SYMBOLS[n] for n in unique_numbers]) + "\n"
    counts = [int(np.sum(np.array(atomic_numbers) == n)) for n in unique_numbers]
    result += "   " + "   ".join(str(c) for c in counts)
    return result + "\n"


def _get_lattice_str(lattice: NDArray[np.float64]) -> str:
    """Return 3-line lattice string."""
    result = ""
    for vector in lattice:
        result += f"     {vector[0]:9.16f}   {vector[1]:9.16f}   {vector[2]:9.16f}\n"
    return result


def _get_positions_str(positions: NDArray[np.float64]) -> str:
    """Return positions string."""
    result = ""
    for vector in positions:
        result += f"  {vector[0]:9.16f}   {vector[1]:9.16f}   {vector[2]:9.16f}\n"
    return result


def write_structure(  # pylint: disable=too-many-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    filepath: str | Path,
    overwrite: bool = False,
    label: str = "POSCAR written by ramannoodle",
) -> None:
    """Write structure to a VASP POSCAR file.

    Parameters
    ----------
    lattice
        | (Å) 2D array with shape (3,3).
    atomic_numbers
        | 1D list of length N where N is the number of atoms.
    positions
        | (fractional) 2D array with shape (N,3).
    filepath
    overwrite
        | Overwrite the file if it exists.
    label
        | POSCAR label (first line).
    """
    verify_structure(lattice, atomic_numbers, positions)
    filepath = pathify(filepath)

    open_mode = "w" if overwrite else "x"
    filepath = pathify(filepath)

    label_str = repr(label)[1:-1] + "\n"  # Raw string with quotes removed
    lattice_str = _get_lattice_str(lattice)
    symbols_str = _get_symbols_str(atomic_numbers)
    positions_str = _get_positions_str(positions)
    with open(filepath, open_mode, encoding="utf-8") as file:
        file.write(label_str)
        file.write("   1.00000000000000" + "\n")
        file.write(lattice_str)
        file.write(symbols_str)
        file.write("Direct\n")
        file.write(positions_str)
