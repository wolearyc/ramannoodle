"""Generic IO functions.

Generic IO functions are somewhat inflexible but are necessary for certain
functionality. Users are strongly encouraged to use IO functions contained in the
code-specific subpackages. For example, IO for VASP POSCAR and OUTCAR files can be
accomplished using :mod:`ramannoodle.io.vasp.poscar` and
:mod:`ramannoodle.io.vasp.outcar` respectively.

"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics._phonon import Phonons
from ramannoodle.dynamics._trajectory import Trajectory
from ramannoodle.structure._reference import ReferenceStructure
from ramannoodle.exceptions import UserError, get_torch_missing_error
import ramannoodle.io.vasp as vasp_io
from ramannoodle.io._utils import pathify_as_list

TORCH_PRESENT = True
try:
    from ramannoodle.dataset.torch._dataset import PolarizabilityDataset
except UserError:
    TORCH_PRESENT = False

# These  map between file formats and appropriate IO functions.
# Keys must be lower case.
_PHONON_READERS = {
    "outcar": vasp_io.outcar.read_phonons,
    "vasprun.xml": vasp_io.vasprun.read_phonons,
}
_TRAJECTORY_READERS = {
    "outcar": vasp_io.outcar.read_trajectory,
    "vasprun.xml": vasp_io.vasprun.read_trajectory,
}
_POSITION_AND_POLARIZABILITY_READERS = {
    "outcar": vasp_io.outcar.read_positions_and_polarizability,
    "vasprun.xml": vasp_io.vasprun.read_positions_and_polarizability,
}
_STRUCTURE_AND_POLARIZABILITY_READERS = {
    "outcar": vasp_io.outcar.read_structure_and_polarizability,
    "vasprun.xml": vasp_io.vasprun.read_structure_and_polarizability,
}
_POLARIZABILITY_DATASET_READERS = {
    "outcar": vasp_io.outcar.read_polarizability_dataset,
    "vasprun.xml": vasp_io.vasprun.read_polarizability_dataset,
}
_POSITION_READERS = {
    "poscar": vasp_io.poscar.read_positions,
    "outcar": vasp_io.outcar.read_positions,
    "xdatcar": vasp_io.poscar.read_positions,
    "vasprun.xml": vasp_io.vasprun.read_positions,
}
_REFERENCE_STRUCTURE_READERS = {
    "poscar": vasp_io.poscar.read_ref_structure,
    "outcar": vasp_io.outcar.read_ref_structure,
    "xdatcar": vasp_io.poscar.read_ref_structure,
    "vasprun.xml": vasp_io.vasprun.read_ref_structure,
}
_STRUCTURE_WRITERS = {
    "poscar": vasp_io.poscar.write_structure,
    "xdatcar": vasp_io.poscar.write_structure,
}
_TRAJECTORY_WRITERS = {"xdatcar": vasp_io.xdatcar.write_trajectory}


def _process_file_format(filepaths: str | Path | list[str] | list[Path]) -> str:
    """Guess file format from filepath(s).

    The implementation is very simple. It simply looks for relevant
    strings in the filename. In the case of multiple files, the first
    file is used to guess the file format.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        Lowercase file_format string.
    """
    filepaths = pathify_as_list(filepaths)
    filename = filepaths[0].name.lower()
    if "outcar" in filename:
        return "outcar"
    if "vasprun.xml" in filename or "vasprun" in filename:
        return "vasprun.xml"
    if "xdatcar" in filename:
        return "xdatcar"
    if "poscar" in filename:
        return "poscar"
    raise ValueError(f"could not guess file format: {filepaths[0]}")


def read_phonons(filepath: str | Path, file_format: str = "auto") -> Phonons:
    """Read phonons from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"vasprun.xml"``, and ``"auto"`` (see :ref:`Supported
        formats`). Not case sensitive.

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _PHONON_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_trajectory(filepath: str | Path, file_format: str = "auto") -> Trajectory:
    """Read molecular dynamics trajectory from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"vasprun.xml"``, and ``"auto"`` (see :ref:`Supported
        formats`). Not case sensitive. Use :func:`.vasp.xdatcar.read_trajectory` to
        read a trajectory from an XDATCAR.

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _TRAJECTORY_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        if file_format == "xdatcar":
            raise ValueError(
                "generic.read_trajectory does not support xdatcar."
            ) from exc
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_positions_and_polarizability(
    filepath: str | Path,
    file_format: str = "auto",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read fractional positions and polarizability from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"vasprun.xml"``, and ``"auto"`` (see :ref:`Supported
        formats`). Not case sensitive.

    Returns
    -------
    :
        0.  positions -- (fractional) Array with shape (N,3) where N is the number of
            atoms.
        #.  polarizability -- Array with shape (3,3).

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _POSITION_AND_POLARIZABILITY_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_structure_and_polarizability(
    filepath: str | Path,
    file_format: str = "auto",
) -> tuple[NDArray[np.float64], list[int], NDArray[np.float64], NDArray[np.float64]]:
    """Read lattice, atomic numbers, fractional positions, polarizability from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"vasprun.xml"``, and ``"auto"`` (see :ref:`Supported
        formats`). Not case sensitive.

    Returns
    -------
    :
        0.  lattice -- (Å) Array with shape (3,3).
        #.  atomic_numbers -- List of length N where N is the number of atoms.
        #.  positions -- (fractional) Array with shape (N,3) where N is the number of
            atoms.
        #.  polarizability -- Array with shape (3,3).

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _STRUCTURE_AND_POLARIZABILITY_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_polarizability_dataset(
    filepaths: str | Path | list[str] | list[Path],
    file_format: str = "auto",
) -> "PolarizabilityDataset":
    """Read polarizability dataset from files.

    Parameters
    ----------
    filepaths
    file_format
        Supports ``"outcar"``, ``"vasprun.xml"``, and ``"auto"`` (see :ref:`Supported
        formats`). Not case sensitive.

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    IncompatibleFileException
        File is incompatible with the dataset.
    """
    if not TORCH_PRESENT:
        raise get_torch_missing_error()
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepaths)
    try:
        return _POLARIZABILITY_DATASET_READERS[file_format.lower()](filepaths)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_positions(
    filepath: str | Path,
    file_format: str = "auto",
) -> NDArray[np.float64]:
    """Read fractional positions from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"poscar"``, ``"xdatcar"``, ``"vasprun.xml"``, and
        ``"auto"`  (see :ref:`Supported formats`). Not case sensitive.

    Returns
    -------
    :
        (fractional) Array with shape (N,3) where N is the number of atoms.

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.

    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _POSITION_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def read_ref_structure(
    filepath: str | Path, file_format: str = "auto"
) -> ReferenceStructure:
    """Read reference structure from a file.

    Parameters
    ----------
    filepath
    file_format
        Supports ``"outcar"``, ``"poscar"``, ``"xdatcar"``, ``"vasprun.xml"``, and
        ``"auto"`` (see :ref:`Supported formats`).

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    SymmetryException
        Structural symmetry determination failed.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        return _REFERENCE_STRUCTURE_READERS[file_format.lower()](filepath)
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


def write_structure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    filepath: str | Path,
    file_format: str = "auto",
    overwrite: bool = False,
) -> None:
    """Write structure to file.

    Parameters
    ----------
    lattice
        (Å) Array with shape (3,3).
    atomic_numbers
        List of length N where N is the number of atoms.
    positions
        (fractional) Array with shape (N,3).
    filepath
    file_format
        Supports ``"poscar"`` and ``"auto"`` (see :ref:`Supported formats`). Not case
        sensitive.
    overwrite
        Overwrite the file if it exists.
    label
        POSCAR label (first line).

    Raises
    ------
    FileExistsError
        File exists and ``overwrite == False``.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        _STRUCTURE_WRITERS[file_format.lower()](
            lattice=lattice,
            atomic_numbers=atomic_numbers,
            positions=positions,
            filepath=filepath,
            overwrite=overwrite,
        )
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc


# pylint: disable=too-many-arguments,too-many-positional-arguments
def write_trajectory(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions_ts: NDArray[np.float64],
    filepath: str | Path,
    file_format: str = "auto",
    overwrite: bool = False,
) -> None:
    """Write trajectory to file.

    Parameters
    ----------
    lattice
        (Å) Array with shape (3,3).
    atomic_numbers
        List of length N where N is the number of atoms.
    positions_ts
        (fractional) Array with shape (S,N,3) where S is the number of
        configurations.
    filepath
    file_format
        Supports ``"xdatcar"`` and ``"auto"`` (see :ref:`Supported formats`). Not case
        sensitive.
    overwrite
        Overwrite the file if it exists.

    Raises
    ------
    FileExistsError
        File exists and ``overwrite == False``.
    """
    if file_format.lower() == "auto":
        file_format = _process_file_format(filepath)
    try:
        _TRAJECTORY_WRITERS[file_format.lower()](
            lattice=lattice,
            atomic_numbers=atomic_numbers,
            positions_ts=positions_ts,
            filepath=filepath,
            overwrite=overwrite,
        )
    except KeyError as exc:
        raise ValueError(f"unsupported format: {file_format}") from exc
