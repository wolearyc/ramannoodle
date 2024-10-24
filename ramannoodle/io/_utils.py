"""Universal IO utility functions."""

from typing import TextIO, Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ramannoodle.exceptions import (
    NoMatchingLineFoundException,
    verify_ndarray_shape,
    verify_positions,
    verify_list_len,
    IncompatibleStructureException,
    get_torch_missing_error,
    UserError,
)
from ramannoodle.constants import ATOM_SYMBOLS

TORCH_PRESENT = True
try:
    from ramannoodle.dataset.torch._dataset import PolarizabilityDataset
except UserError:
    TORCH_PRESENT = False


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
        if len(filepaths) == 0:
            raise ValueError("filepaths is empty")
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


def _read_polarizability_dataset(
    filepaths: str | Path | list[str] | list[Path],
    read_structure_and_polarizability_fn: Callable[
        [str | Path],
        tuple[NDArray[np.float64], list[int], NDArray[np.float64], NDArray[np.float64]],
    ],
) -> "PolarizabilityDataset":
    """Read polarizability dataset from OUTCAR files.

    Parameters
    ----------
    filepath
    read_structure_and_polarizability_fn

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
    InvalidFileException
        Invalid file.
    IncompatibleFileException
        File is incompatible with the dataset.
    """
    if not TORCH_PRESENT:
        raise get_torch_missing_error()
    filepaths = pathify_as_list(filepaths)

    lattice = np.zeros((3, 3))
    atomic_numbers: list[int] = []
    positions_list: list[NDArray[np.float64]] = []
    polarizabilities: list[NDArray[np.float64]] = []
    for file_index, filepath in tqdm(list(enumerate(filepaths)), unit=" files"):
        read_lattice, read_atomic_numbers, positions, polarizability = (
            read_structure_and_polarizability_fn(filepath)
        )
        if file_index != 0:
            if not np.allclose(lattice, read_lattice, atol=1e-5):
                raise IncompatibleStructureException(
                    f"incompatible lattice: {filepath}"
                )
            if atomic_numbers != read_atomic_numbers:
                raise IncompatibleStructureException(
                    f"incompatible atomic numbers: {filepath}"
                )
            if positions_list[0].shape != positions.shape:  # check, just to be safe
                raise IncompatibleStructureException(
                    f"incompatible atomic positions: {filepath}"
                )
        lattice = read_lattice
        atomic_numbers = read_atomic_numbers
        positions_list.append(positions)
        polarizabilities.append(polarizability)

    return PolarizabilityDataset(
        lattice,
        atomic_numbers,
        np.array(positions_list),
        np.array(polarizabilities),
    )
