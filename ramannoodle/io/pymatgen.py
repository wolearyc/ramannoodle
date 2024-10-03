"""Functions for interacting with pymatgen."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    get_pymatgen_missing_error,
    UserError,
    verify_list_len,
    verify_ndarray_shape,
    get_type_error,
    IncompatibleStructureException,
)
from ramannoodle.dynamics._trajectory import Trajectory
from ramannoodle.structure._reference import ReferenceStructure

try:
    from ramannoodle.dataset.torch._dataset import PolarizabilityDataset
except UserError:
    pass

try:
    import pymatgen.core
    import pymatgen.core.trajectory
except ImportError as exc:
    raise get_pymatgen_missing_error() from exc


def _get_lattice(pymatgen_structure: pymatgen.core.Structure) -> NDArray[np.float64]:
    """Get lattice from a pymatgen Structure.

    Parameters
    ----------
    pymatgen_structure

    Returns
    -------
    :
        (Å) Array with shape (3,3).

    """
    try:
        return pymatgen_structure.lattice.matrix
    except AttributeError as exc:
        raise get_type_error(
            "pymatgen_structure", pymatgen_structure, "pymatgen.core.Structure"
        ) from exc


def get_positions(pymatgen_structure: pymatgen.core.Structure) -> NDArray[np.float64]:
    """Read fractional positions from a pymatgen Structure.

    Parameters
    ----------
    pymatgen_structure

    Returns
    -------
    :
        (fractional) Array with shape (N,3) where N is the number of atoms.

    """
    try:
        return pymatgen_structure.frac_coords
    except AttributeError as exc:
        raise get_type_error(
            "pymatgen_structure", pymatgen_structure, "pymatgen.core.Structure"
        ) from exc


def _get_atomic_numbers(pymatgen_structure: pymatgen.core.Structure) -> list[int]:
    """Get atomic numbers from a pymatgen Structure.

    Parameters
    ----------
    pymatgen_structure

    Returns
    -------
    :
        List of length N where N is the number of atoms.

    """
    try:
        return list(pymatgen_structure.atomic_numbers)
    except AttributeError as exc:
        raise get_type_error(
            "pymatgen_structure", pymatgen_structure, "pymatgen.core.Structure"
        ) from exc


def get_structure(
    pymatgen_structure: pymatgen.core.Structure,
) -> tuple[NDArray[np.float64], list[int], NDArray[np.float64]]:
    """Get lattice, positions, and atomic numbers from a pymatgen Structure.

    Parameters
    ----------
    pymatgen_structure

    Returns
    -------
    :
        0.  lattice -- (Å) Array with shape (3,3).
        1.  atomic numbers -- List of length N where N is the number of atoms.
        2.  positions -- (fractional) Array with shape (N,3) where N is the number of
            atoms.
    """
    return (
        _get_lattice(pymatgen_structure),
        _get_atomic_numbers(pymatgen_structure),
        get_positions(pymatgen_structure),
    )


def construct_polarizability_dataset(
    pymatgen_structures: list[pymatgen.core.Structure],
    polarizabilities: NDArray[np.float64],
) -> "PolarizabilityDataset":
    """Create a PolarizabilityDataset from of pymatgen Structures and polarizabilities.

    Parameters
    ----------
    pymatgen_structures
        List of length M.
    polarizabilities
        Array with shape (M,3,3).

    """
    verify_list_len("pymatgen_structures", pymatgen_structures, None)
    verify_ndarray_shape(
        "polarizabilities", polarizabilities, (len(pymatgen_structures), 3, 3)
    )
    lattice, atomic_numbers, _ = get_structure(pymatgen_structures[0])
    positions = np.zeros((len(pymatgen_structures), len(atomic_numbers), 3))
    for i, pymatgen_structure in enumerate(pymatgen_structures):
        if not np.allclose(_get_lattice(pymatgen_structure), lattice, atol=1e-5):
            raise IncompatibleStructureException(
                f"incompatible lattice: pymatgen_structures[{i}]"
            )
        if _get_atomic_numbers(pymatgen_structure) != atomic_numbers:
            raise IncompatibleStructureException(
                f"incompatible atomic numbers: pymatgen_structures[{i}]"
            )
        positions[i] = get_positions(pymatgen_structure)
    return PolarizabilityDataset(lattice, atomic_numbers, positions, polarizabilities)


def construct_ref_structure(
    pymatgen_structure: pymatgen.core.Structure,
) -> ReferenceStructure:
    """Create a ReferenceStructure from a pymatgen Structure.

    Parameters
    ----------
    pymatgen_structure

    """
    return ReferenceStructure(
        _get_atomic_numbers(pymatgen_structure),
        _get_lattice(pymatgen_structure),
        get_positions(pymatgen_structure),
    )


def construct_trajectory(
    pymatgen_trajectory: pymatgen.core.trajectory.Trajectory,
    timestep: float,
) -> Trajectory:
    """Create a Trajectory from a pymatgen Trajectory.

    Parameters
    ----------
    pymatgen_trajectory
    timestep
        (fs)

    """
    try:
        if not pymatgen_trajectory.constant_lattice:
            raise ValueError("pymatgen_trajectory must have a constant lattice")
    except AttributeError as exc:
        raise get_type_error(
            "pymatgen_trajectory",
            pymatgen_trajectory,
            "pymatgen.core.trajectory.Trajectory",
        ) from exc
    return Trajectory(pymatgen_trajectory.coords, timestep)
