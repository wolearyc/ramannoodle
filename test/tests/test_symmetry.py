"""Testing for symmetry-related routines."""

from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.symmetry.symmetry_utils import (
    is_collinear_with_all,
    is_non_collinear_with_all,
    are_collinear,
    is_orthogonal_to_all,
    _get_fractional_positions_permutation_matrix,
    apply_pbc,
    apply_pbc_displacement,
)
from ramannoodle.symmetry import StructuralSymmetry


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
    """Test are_collinear (normal)."""
    assert are_collinear(vector_1, vector_2) == known


@pytest.mark.parametrize(
    "vector_1, vector_2, exception_type, in_reason",
    [
        (
            np.array([-5.0, -5.0, 1.0]),
            [1.0, 1.0, 0.0],
            TypeError,
            "vector_2 should have type ndarray, not list",
        ),
        (
            [0.0, 0.0, -1.0],
            np.array([1.0, 0.0, 0.0]),
            TypeError,
            "vector_1 should have type ndarray, not list",
        ),
        (
            np.array([0.0, 0.0, 6.0, 7.9]),
            np.array([0.0, 0.0, -3.0]),
            ValueError,
            "vector_1 and vector_2 have different lengths: 4 != 3",
        ),
    ],
)
def test_are_collinear_exception(
    vector_1: NDArray[np.float64],
    vector_2: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test are_collinear (exception)."""
    with pytest.raises(exception_type) as error:
        are_collinear(vector_1, vector_2)
    assert in_reason in str(error.value)


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
def test_is_orthogonal_to_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]], known: int
) -> None:
    """Test is_orthogonal_to_all (normal)."""
    assert is_orthogonal_to_all(vector_1, vectors) == known


@pytest.mark.parametrize(
    "vector_1, vectors, known",
    [
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]),
            0,
        ),
        (np.array([1.0, 1.0, 0.0]), np.array([[2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]]), -1),
    ],
)
def test_is_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]], known: int
) -> None:
    """Test is_collinear_with_all (normal)."""
    assert is_collinear_with_all(vector_1, vectors) == known


@pytest.mark.parametrize(
    "vector_1, vectors, known",
    [
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([[0.0, 1.0, 0.0], [80.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]),
            -1,
        ),
        (
            np.array([1.0, 1.0, 0.0]),
            np.array([[2.0, 2.0, 0.0], [-1.0, -1.0, 5.0]]),
            0,
        ),
    ],
)
def test_is_non_collinear_with_all(
    vector_1: NDArray[np.float64], vectors: list[NDArray[np.float64]], known: int
) -> None:
    """Test is_non_collinear_with_all (normal)."""
    assert is_non_collinear_with_all(vector_1, vectors) == known


@pytest.mark.parametrize(
    "outcar_symmetry_fixture, known_nonequivalent_atoms,"
    "known_orthogonal_displacements, known_displacements_shape",
    [
        ("test/data/TiO2/phonons_OUTCAR", 2, 36, [2] * 36),
        ("test/data/STO_RATTLED_OUTCAR", 135, 1, [1]),
        ("test/data/LLZO/LLZO_OUTCAR", 9, 32, [1] * 32),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_structural_symmetry(
    outcar_symmetry_fixture: StructuralSymmetry,
    known_nonequivalent_atoms: int,
    known_orthogonal_displacements: int,
    known_displacements_shape: list[int],
) -> None:
    """Test StructuralSymmetry (normal)."""
    # Equivalent atoms test
    symmetry = outcar_symmetry_fixture
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
    """Test _get_fractional_positions_permutation_matrix (normal)."""
    assert np.isclose(
        _get_fractional_positions_permutation_matrix(reference, permuted), known
    ).all()


@pytest.mark.parametrize(
    "positions, known",
    [
        (np.array([0.2, 0.3, 0]), np.array([0.2, 0.3, 0])),
        (np.array([1.2, 1.3, 1.8]), np.array([0.2, 0.3, 0.8])),
        (np.array([-6.2, -0.3, -0.4]), np.array([0.8, 0.7, 0.6])),
    ],
)
def test_apply_pbc(positions: NDArray[np.float64], known: NDArray[np.float64]) -> None:
    """Test apply_pbc (normal)."""
    assert np.isclose(apply_pbc(positions), known).all()


@pytest.mark.parametrize(
    "displacement, known",
    [
        (np.array([0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4])),
        (np.array([1.8, -0.6, 0]), np.array([-0.2, 0.4, 0])),
        (np.array([-4.51, -0.3, 9.6]), np.array([0.49, -0.3, -0.4])),
    ],
)
def test_apply_pbc_displacement(
    displacement: NDArray[np.float64], known: NDArray[np.float64]
) -> None:
    """Test test_apply_pbc_displacement (normal)."""
    assert np.isclose(apply_pbc_displacement(displacement), known).all()


@pytest.mark.parametrize(
    "atomic_numbers, lattice, fractional_positions, exception_type, in_reason",
    [
        (
            (1, 2, 3, 4),
            np.diag([1, 1, 1]),
            np.zeros((4, 3)),
            TypeError,
            "atomic_numbers should have type ndarray, not tuple",
        ),
        (
            np.array((1, 2, 3, 4)),
            np.diag([1, 1]),
            np.zeros((4, 3)),
            ValueError,
            "lattice has wrong shape: (2,2) != (3,3)",
        ),
        (
            np.array((1, 2, 3, 4)),
            np.diag([1, 1, 1]),
            np.zeros((4, 2)),
            ValueError,
            "fractional_positions has wrong shape: (4,2) != (4,3)",
        ),
        (
            np.array((1, 2, 3, 4)),
            np.diag([1, 1, 1]),
            np.zeros((3, 3)),
            ValueError,
            "fractional_positions has wrong shape: (3,3) != (4,3)",
        ),
    ],
)
def test_structural_symmetry_exception(
    atomic_numbers: NDArray[np.int32],
    lattice: NDArray[np.float64],
    fractional_positions: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test StructuralSymmetry (exception)."""
    with pytest.raises(exception_type) as error:
        StructuralSymmetry(atomic_numbers, lattice, fractional_positions)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture, atom_symbols, known_atom_indexes",
    [
        ("test/data/TiO2/phonons_OUTCAR", "Ti", list(range(0, 36))),
        ("test/data/STO_RATTLED_OUTCAR", "O", list(range(54, 135))),
        ("test/data/LLZO/LLZO_OUTCAR", "La", list(range(56, 80))),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_get_atom_indexes(
    outcar_symmetry_fixture: StructuralSymmetry,
    atom_symbols: str | list[str],
    known_atom_indexes: list[int],
) -> None:
    """Test get_atom_indexes."""
    symmetry = outcar_symmetry_fixture
    assert symmetry.get_atom_indexes(atom_symbols) == known_atom_indexes
