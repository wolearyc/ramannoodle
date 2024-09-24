"""Testing for InterpolationModel."""

from typing import Type
import re

import numpy as np
from numpy.typing import NDArray
import pytest

from ramannoodle.pmodel.interpolation import find_duplicates
from ramannoodle.pmodel.interpolation import InterpolationModel
from ramannoodle.exceptions import InvalidDOFException
from ramannoodle.structure.reference import ReferenceStructure

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
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        find_duplicates(vectors)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,displaced_atom_index, amplitudes, known_dof_added",
    [
        ("test/data/STO_RATTLED_OUTCAR", 0, np.array([-0.05, 0.05, 0.01, -0.01]), 1),
        ("test/data/TiO2/phonons_OUTCAR", 0, np.array([0.01]), 72),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_dof(
    outcar_ref_structure_fixture: ReferenceStructure,
    displaced_atom_index: int,
    amplitudes: NDArray[np.float64],
    known_dof_added: int,
) -> None:
    """Test add_dof (normal)."""
    ref_structure = outcar_ref_structure_fixture
    model = InterpolationModel(ref_structure, np.zeros((3, 3)))
    displacement = ref_structure.positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(amplitudes), 3, 3))
    model.add_dof(displacement, amplitudes, polarizabilities, 1)
    assert len(model.cart_basis_vectors) == known_dof_added
    assert np.isclose(np.linalg.norm(model.cart_basis_vectors[0]), 1)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,displaced_atom_indexes, amplitudes,polarizabilities,"
    "interpolation_order,exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            5,
            InvalidDOFException,
            "insufficient points",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([-0.1, 0.1]),
            np.zeros((1, 3, 3)),
            1,
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([-0.1, 0.1]),
            np.zeros((2, 5, 3)),
            1,
            ValueError,
            "polarizabilities has wrong shape",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0], [0, 1]],
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
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
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            -1.6,
            TypeError,
            "interpolation_order should have type int, not float",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([-0.1, 0.1]),
            np.zeros((2, 3, 3)),
            0,
            ValueError,
            "invalid interpolation_order: 0 < 1",
        ),
        (
            "test/data/STO_RATTLED_OUTCAR",
            [[0]],
            np.array([-0.1, 0.1, 0.1]),
            np.zeros((3, 3, 3)),
            1,
            InvalidDOFException,
            "due to symmetry, amplitude 0.1 should not be specified",
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_dof_exception(
    outcar_ref_structure_fixture: ReferenceStructure,
    displaced_atom_indexes: list[list[int]],
    amplitudes: NDArray[np.float64],
    polarizabilities: NDArray[np.float64],
    interpolation_order: int,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_dof (exception)."""
    ref_structure = outcar_ref_structure_fixture
    model = InterpolationModel(ref_structure, np.zeros((3, 3)))
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        for atom_indexes in displaced_atom_indexes:
            for atom_index in atom_indexes:
                displacement = ref_structure.positions * 0
                displacement[atom_index] = 1
                model.add_dof(
                    displacement, amplitudes, polarizabilities, interpolation_order
                )


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,outcar_file_groups,interpolation_order,"
    "exception_type,in_reason",
    [
        (
            "test/data/STO_RATTLED_OUTCAR",
            [["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"]],
            1,
            InvalidDOFException,
            "incompatible outcar",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"]],
            3,
            InvalidDOFException,
            "insufficient points (3)",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                    "test/data/TiO2/Ti5_0.1x_eps_OUTCAR",
                ]
            ],
            1,
            InvalidDOFException,
            "due to symmetry, amplitude",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                [
                    "this_outcar_does_not_exist",
                ]
            ],
            1,
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
            1,
            InvalidDOFException,
            "is not collinear",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            [
                ["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"],
                ["test/data/TiO2/Ti5_0.1x_eps_OUTCAR"],
            ],
            1,
            InvalidDOFException,
            "new dof is not orthogonal with existing dof (index=0)",
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_add_dof_from_files_exception(
    outcar_ref_structure_fixture: ReferenceStructure,
    outcar_file_groups: list[str],
    interpolation_order: int,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test add_dof_from_files (exception)."""
    ref_structure = outcar_ref_structure_fixture
    model = InterpolationModel(ref_structure, np.zeros((3, 3)))
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        for outcar_files in outcar_file_groups:
            model.add_dof_from_files(outcar_files, "outcar", interpolation_order)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,displaced_atom_index, amplitudes",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            0,
            np.array([0.01]),
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_dummy_interpolation_model(
    outcar_ref_structure_fixture: ReferenceStructure,
    displaced_atom_index: int,
    amplitudes: NDArray[np.float64],
) -> None:
    """Test dummy art models (normal)."""
    ref_structure = outcar_ref_structure_fixture
    model = InterpolationModel(ref_structure, np.zeros((3, 3)), is_dummy_model=True)
    displacement = ref_structure.positions * 0
    displacement[displaced_atom_index][0] = 1.0
    polarizabilities = np.zeros((len(amplitudes), 3, 3))
    model.add_dof(displacement, amplitudes, polarizabilities, 1)

    mask = model.mask
    mask[0] = True
    model.mask = mask

    assert "ATTENTION: this is a dummy model." in repr(model)
    assert "degrees of freedom are masked" in repr(model)

    model.unmask()
    assert not model.mask.all()
