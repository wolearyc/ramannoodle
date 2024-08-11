"""Functions for interacting with VASP POSCAR files."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.io.io_utils import verify_structure, pathify
from ramannoodle.exceptions import InvalidOptionException
from ramannoodle.globals import ATOM_SYMBOLS


def _scrub_options(**options: str) -> dict[str, str]:
    """Check option validity and add default values if necessary."""
    if "label" not in options:
        options["label"] = "POSCAR written by ramannoodle"
    if "overwrite" not in options:
        options["overwrite"] = "false"
    elif "\n" in options["label"]:
        raise InvalidOptionException("option 'label' cannot contain newline")

    allowed_options = ["overwrite", "label"]
    for key in options:
        if key not in allowed_options:
            raise InvalidOptionException(f"'{key}' is not a valid write option")

    return options


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
    result = "Direct \n"
    for vector in positions:
        result += f"  {vector[0]:9.16f}   {vector[1]:9.16f}   {vector[2]:9.16f}\n"
    return result


def _float_formatter(value: np.float64) -> str:
    return f"{float(value):.16f}"


def write_structure(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    filepath: str | Path,
    **options: str,
) -> None:
    """Write structure to a VASP POSCAR file."""
    verify_structure(lattice, atomic_numbers, positions)
    filepath = pathify(filepath)
    options = _scrub_options(**options)

    open_mode = "w" if options["overwrite"] == "true" else "x"
    filepath = pathify(filepath)

    label_str = options["label"] + "\n"
    lattice_str = _get_lattice_str(lattice)
    symbols_str = _get_symbols_str(atomic_numbers)
    positions_str = _get_positions_str(positions)
    with open(filepath, open_mode, encoding="utf-8") as poscar_file:
        poscar_file.write(label_str)
        poscar_file.write("   1.00000000000000" + "\n")
        poscar_file.write(lattice_str)
        poscar_file.write(symbols_str)
        poscar_file.write(positions_str)
