"""Tests for VASP-related functions."""

from pathlib import Path
from typing import Type

import numpy as np
from numpy.typing import NDArray
import pytest

import ramannoodle.io.generic as generic_io
from ramannoodle.constants import ATOMIC_WEIGHTS
from ramannoodle.exceptions import InvalidFileException


@pytest.mark.parametrize(
    "path_fixture, known_num_atoms, known_wavenumbers,"
    "known_first_displacement, known_last_displacement",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            108,
            np.array([811.691808, 811.691808, 811.691808, 811.691808]),
            np.array([-0.00599217, 0.00407925, 0.0]) / np.sqrt(ATOMIC_WEIGHTS["Ti"]),
            np.array([-0.00103298, 0.00651367, 0.0]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
        ),
        (
            "test/data/STO/phonons_OUTCAR",
            319,
            np.array([834.330726, 823.697768, 823.697768, 823.697768]),
            np.array([-1.01496507e-06, 1.01496507e-06, 1.01496507e-06])
            / np.sqrt(ATOMIC_WEIGHTS["Sr"]),
            np.array([0.0, -0.00600504, 0.00012154]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
        ),
    ],
    indirect=["path_fixture"],
)
def test_read_phonons_from_outcar(
    path_fixture: Path,
    known_num_atoms: int,
    known_wavenumbers: NDArray[np.float64],
    known_first_displacement: NDArray[np.float64],
    known_last_displacement: NDArray[np.float64],
) -> None:
    """Test read_phonons for outcar (normal)."""
    phonons = generic_io.read_phonons(path_fixture, file_format="outcar")

    known_degrees_of_freedom = known_num_atoms * 3
    assert phonons.wavenumbers.shape == (known_degrees_of_freedom,)
    assert np.allclose(phonons.wavenumbers[0:4], known_wavenumbers)
    assert phonons.displacements.shape == (
        known_degrees_of_freedom,
        known_num_atoms,
        3,
    )
    assert np.allclose(phonons.displacements[0, 0], known_first_displacement)
    print(phonons.displacements[-1, -1])
    assert np.allclose(phonons.displacements[-1, -1], known_last_displacement)


@pytest.mark.parametrize(
    "path_fixture, exception_type, in_reason",
    [
        ("test/data/TiO2/POSCAR", InvalidFileException, "POTCAR block not found"),
        ("test/data/TiO2/POSCAR", InvalidFileException, "POTCAR block not found"),
    ],
    indirect=["path_fixture"],
)
def test_read_phonons_from_outcar_exception(
    path_fixture: Path, exception_type: Type[Exception], in_reason: str
) -> None:
    """Test read_phonons for outcar (normal)."""
    with pytest.raises(exception_type) as err:
        generic_io.read_phonons(path_fixture, file_format="outcar")
    assert in_reason in str(err.value)


@pytest.mark.parametrize(
    "path_fixture, trajectory_length, last_position",
    [
        (
            "test/data/LLZO/OUTCAR_trajectory",
            15,
            np.array([0.83330583, 0.83331287, 0.29209206]),
        ),
    ],
    indirect=["path_fixture"],
)
def test_read_trajectory_from_outcar(
    path_fixture: Path, trajectory_length: int, last_position: NDArray[np.float64]
) -> None:
    """Test read_trajectory for outcar (normal)."""
    trajectory = generic_io.read_trajectory(path_fixture, file_format="outcar")
    assert len(trajectory) == trajectory_length
    assert np.allclose(last_position, trajectory[-1][-1])
