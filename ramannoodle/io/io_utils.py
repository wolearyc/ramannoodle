"""Universal IO utility functions."""

from typing import TextIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    NoMatchingLineFoundException,
    verify_ndarray_shape,
    verify_positions,
    verify_list_len,
)
from ramannoodle.globals import ATOM_SYMBOLS


def _skip_file_until_line_contains(file: TextIO, content: str) -> str:
    """Read through a file until a line containing specific content is found."""
    for line in file:
        if content in line:
            return line
    raise NoMatchingLineFoundException(content)


def pathify(filepath: str | Path) -> Path:
    """Convert filepath to Path.

    :meta private:
    """
    try:
        return Path(filepath)
    except TypeError as exc:
        raise TypeError(f"{filepath} cannot be resolved as a filepath") from exc


def pathify_as_list(filepaths: str | Path | list[str] | list[Path]) -> list[Path]:
    """Convert filepaths to list of Paths.

    :meta private:
    """
    if isinstance(filepaths, list):
        paths = []
        for item in filepaths:
            paths.append(pathify(item))
        return paths
    return [pathify(filepaths)]


def verify_structure(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
) -> None:
    """Verify a structure.

    :meta private:
    """
    verify_ndarray_shape("lattice", lattice, (3, 3))
    verify_list_len("atomic_numbers", atomic_numbers, None)
    for atomic_number in atomic_numbers:
        if atomic_number not in ATOM_SYMBOLS.keys():
            raise ValueError(f"invalid atomic number: {atomic_number}")
    verify_ndarray_shape("positions", positions, (len(atomic_numbers), 3))
    verify_positions("positions", positions)


def verify_trajectory(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions_ts: NDArray[np.float64],
) -> None:
    """Verify a trajectory.

    :meta private:
    """
    verify_ndarray_shape("lattice", lattice, (3, 3))
    verify_list_len("atomic_numbers", atomic_numbers, None)
    for atomic_number in atomic_numbers:
        if atomic_number not in ATOM_SYMBOLS.keys():
            raise ValueError(f"invalid atomic number: {atomic_number}")
    verify_ndarray_shape("positions_ts", positions_ts, (None, len(atomic_numbers), 3))
    if (0 > positions_ts).any() or (positions_ts > 1.0).any():
        raise ValueError("positions_ts has coordinates that are not between 0 and 1")
