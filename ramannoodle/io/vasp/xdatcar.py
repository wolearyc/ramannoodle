"""Functions for interacting with VASP XDATCAR files."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.io.io_utils import verify_trajectory, pathify
from ramannoodle.exceptions import InvalidFileException
from ramannoodle.dynamics.trajectory import Trajectory
from ramannoodle.io.vasp.poscar import (
    _read_lattice,
    _read_atomic_symbols,
    _read_positions,
    _get_lattice_str,
    _get_symbols_str,
    _get_positions_str,
)


def read_positions_ts(
    filepath: str | Path,
) -> NDArray[np.float64]:
    """Read fractional positions time series from a VASP XDATCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        (fractional) 2D array with shape (S,N,3) where S is the number of configurations
        and N is the number of atoms.

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    filepath = pathify(filepath)
    positions_ts = []
    with open(filepath, "r", encoding="utf-8") as file:
        lattice = _read_lattice(file)
        atomic_symbols = _read_atomic_symbols(file)
        while True:
            try:
                positions = _read_positions(file, lattice, len(atomic_symbols))
                positions_ts.append(positions)
            except InvalidFileException as exc:
                if "missing first character in coordinate format:" in str(exc):
                    break
                raise exc
        return np.array(positions_ts)


def read_trajectory(
    filepath: str | Path,
    timestep: float,
) -> Trajectory:
    """Read trajectory from a VASP XDATCAR file.

    Timestep must be manually specified, as XDATCAR's do not contain the timestep.

    Parameters
    ----------
    filepath
    timestep
        | (fs)

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    positions_ts = read_positions_ts(filepath)
    return Trajectory(positions_ts, timestep)


def write_trajectory(  # pylint: disable=too-many-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions_ts: NDArray[np.float64],
    filepath: str | Path,
    overwrite: bool = False,
    label: str = "XDATCAR written by ramannoodle",
) -> None:
    """Write trajectory to a VASP XDATCAR file.

    Parameters
    ----------
    lattice
        | (Å) 2D array with shape (3,3).
    atomic_numbers
        | 1D list of length N where N is the number of atoms.
    positions_ts
        | (fractional) 3D array with shape (S,N,3) where S is the number of
        | configurations.
    filepath
    overwrite
        | Overwrite the file if it exists.
    label
        | XDATCAR label (first line).
    """
    verify_trajectory(lattice, atomic_numbers, positions_ts)
    filepath = pathify(filepath)

    open_mode = "w" if overwrite else "x"
    filepath = pathify(filepath)

    label_str = repr(label)[1:-1] + "\n"  # Raw string with quotes removed
    lattice_str = _get_lattice_str(lattice)
    symbols_str = _get_symbols_str(atomic_numbers)
    with open(filepath, open_mode, encoding="utf-8") as file:
        file.write(label_str)
        file.write("   1.00000000000000" + "\n")
        file.write(lattice_str)
        file.write(symbols_str)
        for index, positions in enumerate(positions_ts):
            config_label = f"Direct configuration= {index+1:>6}\n"
            positions_str = _get_positions_str(positions)
            file.write(config_label)
            file.write(positions_str)
