"""Tests for VASP-related routines."""

from pathlib import Path
from typing import Type

import numpy as np
from numpy.typing import NDArray
import pytest

import ramannoodle.io.generic as generic_io
from ramannoodle.globals import ATOMIC_WEIGHTS
from ramannoodle.exceptions import InvalidFileException


@pytest.mark.parametrize(
    "path_fixture, known_num_atoms, known_wavenumbers,"
    "known_first_displacement, known_last_displacement",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            108,
            np.array([811.691808, 811.691808, 811.691808, 811.691808]),
            np.array([-0.068172, 0.046409, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["Ti"]),
            np.array([-0.011752, 0.074105, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
        ),
        (
            "test/data/STO/phonons_OUTCAR",
            319,
            np.array([834.330726, 823.697768, 823.697768, 823.697768]),
            np.array([-0.000016, 0.000016, 0.000016]) / np.sqrt(ATOMIC_WEIGHTS["Sr"]),
            np.array([0.000000, -0.094664, 0.001916]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
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
    assert np.isclose(phonons.wavenumbers[0:4], known_wavenumbers).all()
    assert phonons.cart_displacements.shape == (
        known_degrees_of_freedom,
        known_num_atoms,
        3,
    )
    assert np.isclose(phonons.cart_displacements[0, 0], known_first_displacement).all()
    print(phonons.cart_displacements[-1, -1])
    assert np.isclose(phonons.cart_displacements[-1, -1], known_last_displacement).all()


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
