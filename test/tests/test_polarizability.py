"""Testing for the polarizability."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.polarizability_utils import find_duplicates
from ramannoodle.polarizability.interpolation import InterpolationPolarizabilityModel
from ramannoodle.io.vasp import read_structural_symmetry_from_outcar
from ramannoodle.exceptions import InvalidDOFException
from ramannoodle.symmetry import StructuralSymmetry
from ramannoodle import io

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
        ("test/data/TiO2/PHONON_OUTCAR", 0, np.array([0.01]), 72),
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
        ("test/data/TiO2/PHONON_OUTCAR", 0, np.array([-0.01, 0.01])),
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


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,ref_eps,to_add,additional_references",
    [
        (
            "test/data/TiO2/PHONON_OUTCAR",
            "test/data/TiO2/ref_OUTCAR",
            [
                ["test/data/TiO2/Ti5_0.1z_OUTCAR", "test/data/TiO2/Ti5_0.2z_OUTCAR"],
                ["test/data/TiO2/Ti5_0.1x_OUTCAR", "test/data/TiO2/Ti5_0.2x_OUTCAR"],
                [
                    "test/data/TiO2/O43_0.1z_OUTCAR",
                    "test/data/TiO2/O43_0.2z_OUTCAR",
                    "test/data/TiO2/O43_m0.1z_OUTCAR",
                    "test/data/TiO2/O43_m0.2z_OUTCAR",
                ],
                [
                    "test/data/TiO2/O43_0.1x_OUTCAR",
                    "test/data/TiO2/O43_0.2x_OUTCAR",
                ],
                [
                    "test/data/TiO2/O43_0.1y_OUTCAR",
                    "test/data/TiO2/O43_0.2y_OUTCAR",
                ],
            ],
            [
                "test/data/TiO2/Ti5_m0.2z_OUTCAR",
                "test/data/TiO2/Ti5_m0.1z_OUTCAR",
                "test/data/TiO2/O43_m0.1x_OUTCAR",
                "test/data/TiO2/O43_m0.2x_OUTCAR",
                "test/data/TiO2/O43_m0.1y_OUTCAR",
                "test/data/TiO2/O43_m0.2y_OUTCAR",
            ],
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_get_polarizability(
    outcar_symmetry_fixture: StructuralSymmetry,
    ref_eps: str,
    to_add: list[str],
    additional_references: list[str],
) -> None:
    """Test."""
    symmetry = outcar_symmetry_fixture
    _, polarizability = io.read_positions_and_polarizability(
        ref_eps, file_format="outcar"
    )
    model = InterpolationPolarizabilityModel(symmetry, polarizability)
    for outcar_path_list in to_add:
        model.add_dof_from_files(
            outcar_path_list, file_format="outcar", interpolation_order=2
        )

    # Tests
    total_outcar_paths = additional_references
    for outcar_path in to_add:
        total_outcar_paths += outcar_path
    for outcar_path in total_outcar_paths:
        positions, known_polarizability = io.read_positions_and_polarizability(
            outcar_path, file_format="outcar"
        )
        cartesian_displacement = symmetry.get_cartesian_displacement(
            positions - symmetry.get_fractional_positions()
        )
        model_polarizability = model.get_polarizability(cartesian_displacement)
        assert np.isclose(model_polarizability, known_polarizability, atol=1e-4).all()
