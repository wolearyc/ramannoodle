"""Testing for the polarizability."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import find_duplicates
from ramannoodle.polarizability.interpolation import InterpolationPolarizabilityModel
from ramannoodle.io.vasp import read_structural_symmetry_from_outcar
from ramannoodle.exceptions import InvalidDOFException

# pylint: disable=protected-access


@pytest.mark.parametrize(
    "vectors, known",
    [
        (np.array([-0.05, 0.05, 0.01, -0.01]), None),
        (np.array([-0.05, 0.05, -0.05, -0.01]), -0.05),
    ],
)
def test_find_duplicates(vectors: list[NDArray[np.float64]], known: bool) -> None:
    """Test."""
    assert find_duplicates(vectors) == known


@pytest.mark.parametrize(
    "outcar_path_fixture,displaced_atom_index, amplitudes, known_dof_added",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([-0.05, 0.05, 0.01, -0.01]), 1),
        ("test/data/TiO2/phonons_OUTCAR", 0, np.array([0.01]), 72),
    ],
    indirect=["outcar_path_fixture"],
)
def test_add_dof(
    outcar_path_fixture: Path,
    displaced_atom_index: int,
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test."""
    symmetry = read_structural_symmetry_from_outcar(outcar_path_fixture)
    model = InterpolationPolarizabilityModel(symmetry, np.zeros((3, 3)))
    displacement = symmetry._fractional_positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(amplitudes), 3, 3))
    model.add_dof(displacement, amplitudes, polarizabilities, 1)
    assert len(model._cartesian_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model._cartesian_basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_path_fixture,displaced_atom_index, amplitudes",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([0.01, 0.01])),
        ("test/data/TiO2/phonons_OUTCAR", 0, np.array([-0.01, 0.01])),
    ],
    indirect=["outcar_path_fixture"],
)
def test_overspecified_dof(
    outcar_path_fixture: Path,
    displaced_atom_index: int,
    amplitudes: NDArray[np.float64],
) -> None:
    """Test."""
    symmetry = read_structural_symmetry_from_outcar(outcar_path_fixture)
    model = InterpolationPolarizabilityModel(symmetry, np.zeros((3, 3)))
    displacement = symmetry._fractional_positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(amplitudes), 3, 3))
    with pytest.raises(InvalidDOFException) as error:
        model.add_dof(displacement, amplitudes, polarizabilities, 1)

    assert "should not be specified" in str(error.value)