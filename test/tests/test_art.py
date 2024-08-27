"""Testing for ARTModel."""

from typing import Type

import numpy as np
from numpy.typing import NDArray
import pytest

from ramannoodle.polarizability.art import ARTModel
from ramannoodle.exceptions import InvalidDOFException, UsageError
from ramannoodle.structure.reference import ReferenceStructure

# pylint: disable=protected-access
# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_index, cart_direction, amplitudes,"
    "known_dof_added",
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
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_art(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test add_art (normal)."""
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)))
    model.add_art(
        atom_index, cart_direction, amplitudes, np.zeros((amplitudes.size, 3, 3))
    )
    assert len(model.cart_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model.cart_basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_indexes,cart_directions,amplitudes,"
    "polarizabilities,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",  # This case gives a warning.
            [0],
            np.array([[1, 0, 0]]),
            np.array([0.1]),
            np.zeros((1, 3, 3)),
            InvalidDOFException,
            "insufficient points",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",  # This case gives a warning.
            [0],
            np.array([[1, 0]]),
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            ValueError,
            "direction has wrong shape: (2,) != (3,)",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [0],
            np.array([[1, 0, 0]]),
            np.array([]),
            np.zeros((1, 3, 3)),
            ValueError,
            "amplitudes has wrong shape: (0,) != (1,) or (2,)",
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
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_art_exception(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_indexes: list[int],
    cart_directions: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    polarizabilities: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_art (exception)."""
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        for atom_index, direction in zip(atom_indexes, cart_directions):
            model.add_art(atom_index, direction, amplitudes, polarizabilities)

    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,outcar_file_groups,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            [["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"]],
            InvalidDOFException,
            "incompatible outcar",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                    "test/data/TiO2/Ti5_0.2x_eps_OUTCAR",
                ]
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 4 != 2",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                    "test/data/TiO2/Ti5_m0.1x_eps_OUTCAR",
                ]
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 4 != 2",
        ),
        (
            "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
            [
                [
                    "test/data/TiO2/O43_0.1x_eps_OUTCAR",
                ]
            ],
            InvalidDOFException,
            "multiple atoms displaced simultaneously",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "this_outcar_does_not_exist",
                ]
            ],
            FileNotFoundError,
            "No such file or directory",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                    "test/data/TiO2/Ti5_0.1y_eps_OUTCAR",
                ]
            ],
            InvalidDOFException,
            "is not collinear",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/O43_0.1z_eps_OUTCAR",
                ]
            ],
            InvalidDOFException,
            "wrong number of amplitudes: 1 != 2",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                ],
                [
                    "test/data/TiO2/Ti5_0.1y_eps_OUTCAR",
                ],
            ],
            InvalidDOFException,
            "is not orthogonal",
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_art_from_files_exception(
    outcar_ref_structure_fixture: ReferenceStructure,
    outcar_file_groups: list[str],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_art_from_files (exception)."""
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)))
    with pytest.raises(exception_type) as error:
        for outcar_files in outcar_file_groups:
            model.add_art_from_files(outcar_files, "outcar")
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_index, cart_direction, amplitudes, "
    "known_tuples_len,known_atom_index, known_directions, known_equivalent_atoms",
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
    indirect=["outcar_ref_structure_fixture"],
)
def test_get_specification_tuples(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    known_tuples_len: int,
    known_atom_index: int,
    known_directions: list[NDArray[np.float64]],
    known_equivalent_atoms: list[int],
) -> None:
    """Test get_specification_tuples."""
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)))
    model.add_art(
        atom_index, cart_direction, amplitudes, np.zeros((amplitudes.size, 3, 3))
    )
    specification_tuples = model.get_specification_tuples()

    assert len(specification_tuples) == known_tuples_len
    assert specification_tuples[0][0] == known_atom_index
    assert np.isclose(
        specification_tuples[0][2],
        known_directions,
        atol=1e-7,
    ).all()
    assert specification_tuples[0][1] == known_equivalent_atoms


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_index, cart_direction, amplitudes,"
    "known_dof_added",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            0,
            np.array([1, 0, 0]),
            np.array([0.01]),
            72,
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_dummy_art(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test dummy art models (normal)."""
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)), is_dummy_model=True)
    model.add_art(
        atom_index, cart_direction, amplitudes, np.zeros((amplitudes.size, 3, 3))
    )
    mask = model.mask
    mask[0] = True
    model.mask = mask

    assert "ATTENTION: this is a dummy model." in repr(model)
    assert "atomic Raman tensors are masked" in repr(model)
    assert len(model.cart_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model.cart_basis_vectors[0]), 1)
    with pytest.raises(UsageError) as err:
        model.add_dof(np.array([]), np.array([]), np.array([]), 1, False)
    assert "add_dof should not be used; use add_art instead" in str(err.value)
    with pytest.raises(UsageError) as err:
        model.add_dof_from_files(["blah"], "blah", 1)
    assert (
        "add_dof_from_files should not be used; use add_art_from_files instead"
        in str(err.value)
    )
    with pytest.raises(UsageError) as err:
        model.calc_polarizability(np.array([]))
    assert "dummy model cannot calculate polarizabilities" in str(err.value)
