"""Tests for VASP-related routines."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.generic as generic_io
from ramannoodle.globals import ATOMIC_WEIGHTS


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
    assert phonons.cartesian_displacements.shape == (
        known_degrees_of_freedom,
        known_num_atoms,
        3,
    )
    assert np.isclose(
        phonons.cartesian_displacements[0, 0], known_first_displacement
    ).all()
    print(phonons.cartesian_displacements[-1, -1])
    assert np.isclose(
        phonons.cartesian_displacements[-1, -1], known_last_displacement
    ).all()
