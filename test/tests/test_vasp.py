"""Tests for VASP-related routines."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.io import (
    load_phonons,
)
from ramannoodle.globals import ATOMIC_WEIGHTS

from .. import PHONONS_OUTCAR_NUM_ATOMS


@pytest.mark.parametrize(
    "outcar_path_fixture, known_num_atoms, known_wavenumbers,"
    "known_first_displacement, known_last_displacement",
    [
        (
            "test/data/TiO2/PHONON_OUTCAR",
            PHONONS_OUTCAR_NUM_ATOMS,
            np.array([811.691808, 811.691808, 811.691808, 811.691808]),
            np.array([-0.068172, 0.046409, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["Ti"]),
            np.array([-0.011752, 0.074105, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
        ),
    ],
    indirect=["outcar_path_fixture"],
)
def test_load_phonons_from_outcar(
    outcar_path_fixture: Path,
    known_num_atoms: int,
    known_wavenumbers: NDArray[np.float64],
    known_first_displacement: NDArray[np.float64],
    known_last_displacement: NDArray[np.float64],
) -> None:
    """Test."""
    phonons = load_phonons(outcar_path_fixture, file_format="outcar")

    known_degrees_of_freedom = known_num_atoms * 3
    assert phonons.get_wavenumbers().shape == (known_degrees_of_freedom,)
    assert np.isclose(phonons.get_wavenumbers()[0:4], known_wavenumbers).all()
    assert phonons.get_cartesian_displacements().shape == (
        known_degrees_of_freedom,
        known_num_atoms,
        3,
    )
    assert np.isclose(
        phonons.get_cartesian_displacements()[0, 0], known_first_displacement
    ).all()
    print(phonons.get_cartesian_displacements()[-1, -1])
    assert np.isclose(
        phonons.get_cartesian_displacements()[-1, -1], known_last_displacement
    ).all()
