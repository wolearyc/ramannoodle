"""Testing for the polarizability."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import find_duplicates
from ramannoodle.polarizability import InterpolationPolarizabilityModel
from ramannoodle.io.vasp import load_structural_symmetry_from_outcar
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
    "outcar_path_fixture,displaced_atom_index, magnitudes, known_dof_added",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([-0.05, 0.05, 0.01, -0.01]), 1),
        ("test/data/TiO2_OUTCAR", 0, np.array([0.01]), 72),
    ],
    indirect=["outcar_path_fixture"],
)
def test_add_dof(
    outcar_path_fixture: Path,
    displaced_atom_index: int,
    magnitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test."""
    symmetry = load_structural_symmetry_from_outcar(outcar_path_fixture)
    model = InterpolationPolarizabilityModel(symmetry)
    displacement = symmetry._fractional_positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(magnitudes), 3, 3))
    model.add_dof(displacement, magnitudes, polarizabilities, 1)
    assert len(model._basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model._basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_path_fixture,displaced_atom_index, magnitudes",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([0.01, 0.01])),
        ("test/data/TiO2_OUTCAR", 0, np.array([-0.01, 0.01])),
    ],
    indirect=["outcar_path_fixture"],
)
def test_overspecified_dof(
    outcar_path_fixture: Path,
    displaced_atom_index: int,
    magnitudes: NDArray[np.float64],
) -> None:
    """Test."""
    symmetry = load_structural_symmetry_from_outcar(outcar_path_fixture)
    model = InterpolationPolarizabilityModel(symmetry)
    displacement = symmetry._fractional_positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(magnitudes), 3, 3))
    with pytest.raises(InvalidDOFException) as error:
        model.add_dof(displacement, magnitudes, polarizabilities, 1)

    assert "should not be specified" in str(error.value)
