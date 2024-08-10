"""Testing for ARTModel."""

from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.art import ARTModel
from ramannoodle.exceptions import InvalidDOFException
from ramannoodle.symmetry import StructuralSymmetry

# pylint: disable=protected-access
# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,atom_index, direction, amplitudes, known_dof_added",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            0,
            np.array([1, 0, 0]),
            np.array([-0.01, 0.01]),
            1,
        ),
        ("test/data/TiO2/phonons_OUTCAR", 0, np.array([1, 0, 0]), np.array([0.01]), 72),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_art(
    outcar_symmetry_fixture: StructuralSymmetry,
    atom_index: int,
    direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test add_art (normal)."""
    symmetry = outcar_symmetry_fixture
    model = ARTModel(symmetry, np.zeros((3, 3)))
    model.add_art(atom_index, direction, amplitudes, np.zeros((amplitudes.size, 3, 3)))
    assert len(model._cartesian_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model._cartesian_basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,atom_indexes,directions,amplitudes,polarizabilities,"
    "exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0],
            np.array([[1, 0, 0]]),
            np.array([0.1]),
            np.zeros((1, 3, 3)),
            InvalidDOFException,
            "insufficient points",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0],
            np.array([[1, 0, 0]]),
            np.array([-0.1, 0.1]),
            np.zeros((2, 4, 3)),
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0],
            np.array([[1, 0, 0]]),
            np.array([-0.1, 0.1]),
            np.zeros((4, 3, 3)),
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0, 0],
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            InvalidDOFException,
            "not orthogonal",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0, 0],
            np.array([[1, 0, 0], [1, 0, 0]]),
            [-0.1, 0.1],
            np.zeros((2, 3, 3)),
            TypeError,
            "amplitudes should have type ndarray, not list",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0, 0],
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([-0.1, 0.1]),
            list(np.zeros((2, 3, 3))),
            TypeError,
            "polarizabilities should have type ndarray, not list",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0, 0]],
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            TypeError,
            "atom_index should have type int, not list",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0],
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([0.1, 0.1]),
            np.zeros((2, 3, 3)),
            InvalidDOFException,
            "due to symmetry, amplitude 0.1 should not be specified",
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_art_exception(
    outcar_symmetry_fixture: StructuralSymmetry,
    atom_indexes: list[int],
    directions: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    polarizabilities: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_art (exception)."""
    symmetry = outcar_symmetry_fixture
    model = ARTModel(symmetry, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        for atom_index, direction in zip(atom_indexes, directions):
            model.add_art(atom_index, direction, amplitudes, polarizabilities)

    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,outcar_files,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            ["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"],
            InvalidDOFException,
            "incompatible outcar",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                "test/data/TiO2/Ti5_0.2x_eps_OUTCAR",
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 4 != 2",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                "test/data/TiO2/Ti5_m0.1x_eps_OUTCAR",
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 4 != 2",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "this_outcar_does_not_exist",
            ],
            FileNotFoundError,
            "No such file or directory",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                "test/data/TiO2/Ti5_0.1y_eps_OUTCAR",
            ],
            InvalidDOFException,
            "is not collinear",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                "test/data/TiO2/O43_0.1z_eps_OUTCAR",
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 1 != 2",
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_add_art_from_files_exception(
    outcar_symmetry_fixture: StructuralSymmetry,
    outcar_files: list[str],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_dof_from_files (exception)."""
    symmetry = outcar_symmetry_fixture
    model = ARTModel(symmetry, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        model.add_art_from_files(outcar_files, "outcar")
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,atom_index, direction, amplitudes, known_dict_len,"
    "known_atom_index, known_directions, known_equivalent_atoms",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            0,
            np.array([1, 0, 0]),
            np.array([-0.01, 0.01]),
            135,
            0,
            np.array([[1, 0, 0]]),
            [],
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            0,
            np.array([1, 0, 0]),
            np.array([0.01]),
            2,
            0,
            np.array([[1, 0, 0], [0, -1, 0]]),
            list(range(1, 36)),
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_get_specification_dict(
    outcar_symmetry_fixture: StructuralSymmetry,
    atom_index: int,
    direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    known_dict_len: int,
    known_atom_index: int,
    known_directions: list[NDArray[np.float64]],
    known_equivalent_atoms: list[int],
) -> None:
    """Test get_specification_dict."""
    symmetry = outcar_symmetry_fixture
    model = ARTModel(symmetry, np.zeros((3, 3)))
    model.add_art(atom_index, direction, amplitudes, np.zeros((amplitudes.size, 3, 3)))
    status_dict = model.get_specification_dict()

    assert len(status_dict) == known_dict_len
    assert np.isclose(
        status_dict[known_atom_index]["specified_directions"],
        known_directions,
        atol=1e-7,
    ).all()
    assert status_dict[known_atom_index]["equivalent_atoms"] == known_equivalent_atoms
