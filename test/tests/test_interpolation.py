"""Testing for the polarizability."""

from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import find_duplicates
from ramannoodle.polarizability.interpolation import InterpolationModel
from ramannoodle.exceptions import InvalidDOFException
from ramannoodle.symmetry import StructuralSymmetry

# pylint: disable=protected-access
# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    "vectors, known",
    [
        (np.array([-0.05, 0.05, 0.01, -0.01]), None),
        (np.array([-0.05, 0.05, -0.05, -0.01]), -0.05),
    ],
)
def test_find_duplicates(vectors: NDArray[np.float64], known: bool) -> None:
    """Test find_duplicates (normal)."""
    assert find_duplicates(vectors) == known


@pytest.mark.parametrize(
    "vectors, exception_type, in_reason",
    [
        (1, TypeError, "should have type Iterable, not int"),
        (["string", TypeError, "not array_like"]),
    ],
)
def test_find_duplicates_exception(
    vectors: NDArray[np.float64], exception_type: Type[Exception], in_reason: str
) -> None:
    """Test find_duplicates (exception)."""
    with pytest.raises(exception_type) as error:
        find_duplicates(vectors)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,displaced_atom_index, amplitudes, known_dof_added",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([-0.05, 0.05, 0.01, -0.01]), 1),
        ("test/data/TiO2/phonons_OUTCAR", 0, np.array([0.01]), 72),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_dof(
    outcar_symmetry_fixture: StructuralSymmetry,
    displaced_atom_index: int,
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test add_dof (normal)."""
    symmetry = outcar_symmetry_fixture
    model = InterpolationModel(symmetry, np.zeros((3, 3)))
    displacement = symmetry._fractional_positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(amplitudes), 3, 3))
    model.add_dof(displacement, amplitudes, polarizabilities, 1)
    assert len(model._cartesian_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model._cartesian_basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,displaced_atom_indexes, amplitudes,polarizabilities,"
    "interpolation_order,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            np.zeros((1, 3, 3)),
            4,
            InvalidDOFException,
            "insufficient points",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1, 0.1]),
            np.zeros((1, 3, 3)),
            1,
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            np.zeros((2, 3, 3)),
            1,
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            np.zeros((2, 3, 3)),
            1,
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0], [0, 1]],
            np.array([0.1]),
            np.zeros((1, 3, 3)),
            1,
            InvalidDOFException,
            "not orthogonal",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            [0.1],
            np.zeros((1, 3, 3)),
            1,
            TypeError,
            "amplitudes should have type ndarray, not list",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            list(np.zeros((1, 3, 3))),
            1,
            TypeError,
            "polarizabilities should have type ndarray, not list",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            np.zeros((1, 2, 2)),
            1,
            ValueError,
            "polarizabilities has wrong shape: (1,2,2) != (1,3,3)",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1]),
            np.zeros((1, 3, 3)),
            -1.6,
            TypeError,
            "interpolation_order should have type int, not float",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([0.1, 0.1]),
            np.zeros((2, 3, 3)),
            -1.6,
            InvalidDOFException,
            "due to symmetry, amplitude 0.1 should not be specified",
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_dof_exception(
    outcar_symmetry_fixture: StructuralSymmetry,
    displaced_atom_indexes: list[list[int]],
    amplitudes: NDArray[np.float64],
    polarizabilities: NDArray[np.float64],
    interpolation_order: int,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_dof (exception)."""
    symmetry = outcar_symmetry_fixture
    model = InterpolationModel(symmetry, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        for atom_indexes in displaced_atom_indexes:
            for atom_index in atom_indexes:
                displacement = symmetry.get_fractional_positions() * 0
                displacement[atom_index] = 1
                model.add_dof(
                    displacement, amplitudes, polarizabilities, interpolation_order
                )

    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,outcar_files,interpolation_order,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            ["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"],
            1,
            InvalidDOFException,
            "incompatible outcar",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            ["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"],
            3,
            InvalidDOFException,
            "insufficient points (3)",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
            ],
            1,
            InvalidDOFException,
            "due to symmetry, amplitude",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "this_outcar_does_not_exist",
            ],
            1,
            FileNotFoundError,
            "No such file or directory",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                "test/data/TiO2/Ti5_0.1y_eps_OUTCAR",
            ],
            1,
            InvalidDOFException,
            "is not collinear",
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_dof_from_files_exception(
    outcar_symmetry_fixture: StructuralSymmetry,
    outcar_files: list[str],
    interpolation_order: int,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_dof_from_files (exception)."""
    symmetry = outcar_symmetry_fixture
    model = InterpolationModel(symmetry, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        model.add_dof_from_files(outcar_files, "outcar", interpolation_order)
    assert in_reason in str(error.value)
