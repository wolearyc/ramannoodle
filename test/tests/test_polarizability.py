"""Testing for the polarizability."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import find_duplicates
from ramannoodle.symmetry.symmetry_utils import (
    are_collinear,
    is_orthogonal_to_all,
    get_fractional_positions_permutation_matrix,
)
from ramannoodle.polarizability import InterpolationPolarizabilityModel
from ramannoodle.io.vasp import load_structural_symmetry_from_outcar


@pytest.mark.parametrize(
    "vector_1, vector_2, known",
    [
        (np.array([-5.0, -5.0, 1.0]), np.array([1.0, 1.0, 0.0]), False),
        (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), False),
        (np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, -3.0]), True),
        (np.array([0.0, 0.0, -1.0]), np.array([1.0, 3.0, 1.0]), False),
    ],
)
def test_are_collinear(
    vector_1: NDArray[np.float64], vector_2: NDArray[np.float64], known: bool
) -> None:
    """Test"""
    assert are_collinear(vector_1, vector_2) == known


@pytest.mark.parametrize(
    "vector_1, vectors, known",
    [
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]),
            2,
        ),
        (np.array([1.0, 1.0, 0.0]), np.array([[0.0, 0.0, 1.0], [-1.0, 1.0, 0.0]]), -1),
    ],
)
def test_check_orthogonal(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]], known: int
) -> None:
    """Test"""
    assert is_orthogonal_to_all(vector_1, vectors) == known


@pytest.mark.parametrize(
    "outcar_path_fixture, known_nonequivalent_atoms,"
    "known_orthogonal_displacements, known_displacements_shape",
    [
        ("test/data/TiO2_OUTCAR", 2, 36, [2] * 36),
        ("test/data/STO_RATTLED_OUTCAR", 135, 1, [1]),
        ("test/data/LLZO_OUTCAR", 9, 32, [1] * 32),
    ],
    indirect=["outcar_path_fixture"],
)
def test_structural_symmetry(
    outcar_path_fixture: Path,
    known_nonequivalent_atoms: int,
    known_orthogonal_displacements: int,
    known_displacements_shape: list[int],
) -> None:
    """Test"""

    # Equivalent atoms test
    symmetry = load_structural_symmetry_from_outcar(outcar_path_fixture)
    assert symmetry.get_num_nonequivalent_atoms() == known_nonequivalent_atoms

    # Equivalent displacement test
    displacement = (
        symmetry._fractional_positions * 0  # pylint: disable=protected-access
    )
    displacement[0, 2] += 0.1
    print(displacement.shape)
    displacements = symmetry.get_equivalent_displacements(displacement)
    assert len(displacements) == known_orthogonal_displacements
    assert [len(d["displacements"]) for d in displacements] == known_displacements_shape


@pytest.mark.parametrize(
    "reference, permuted, known",
    [
        (
            np.array(
                [[0.2, 0.3, 0.4], [0.2, 0.8, 0.9], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            np.array(
                [[0.2, 0.3, 0.4], [0.0, 0.0, 1.0], [0.2, 0.8, 0.9]], dtype=np.float64
            ),
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
        )
    ],
)
def test_get_fractional_positions_permutation_matrix(
    reference: NDArray[np.float64],
    permuted: NDArray[np.float64],
    known: NDArray[np.float64],
) -> None:
    """test"""
    assert np.isclose(
        get_fractional_positions_permutation_matrix(reference, permuted), known
    ).all()


@pytest.mark.parametrize(
    "vectors, known",
    [
        (np.array([-0.05, 0.05, 0.01, -0.01]), None),
        (np.array([-0.05, 0.05, -0.05, -0.01]), -0.05),
    ],
)
def test_find_duplicates(vectors: list[NDArray[np.float64]], known: bool) -> None:
    """test"""
    assert find_duplicates(vectors) == known


@pytest.mark.parametrize(
    "outcar_path_fixture,displaced_atom_index, magnitudes,known_dof_added",
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
    """test"""
    symmetry = load_structural_symmetry_from_outcar(outcar_path_fixture)
    model = InterpolationPolarizabilityModel(symmetry)
    displacement = (
        symmetry._fractional_positions * 0  # pylint: disable=protected-access
    )
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(magnitudes), 3, 3))
    model.add_dof(displacement, magnitudes, polarizabilities, 1)
    assert (
        len(model._basis_vectors) == known_dof_added  # pylint: disable=protected-access
    )
