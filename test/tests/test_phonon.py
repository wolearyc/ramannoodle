"""Tests for phonon-related functions."""

from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.generic
from ramannoodle.dynamics.phonon import Phonons
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.polarizability.art import ARTModel


@pytest.mark.parametrize(
    "ref_positions, wavenumbers, displacements, exception_type, in_reason",
    [
        (
            {},
            np.array([0, 1, 2, 3]),
            np.zeros((4, 10, 3)),
            TypeError,
            "ref_positions should have type ndarray, not dict",
        ),
        (
            np.random.random([10, 3]),
            [0, 1, 2, 3],
            np.zeros((4, 10, 3)),
            TypeError,
            "wavenumbers should have type ndarray, not list",
        ),
        (
            np.random.random([10, 3]),
            np.array([0, 1, 2, 3]),
            np.zeros((5, 10, 3)),
            ValueError,
            "displacements has wrong shape: (5,10,3) != (4,10,3)",
        ),
    ],
)
def test_phonons_exception(
    ref_positions: NDArray[np.float64],
    wavenumbers: NDArray[np.float64],
    displacements: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test phonon construction (exception)."""
    with pytest.raises(exception_type) as err:
        Phonons(ref_positions, wavenumbers, displacements)
    assert in_reason in str(err.value)


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,data_directory,dof_eps_outcars",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            "test/data/TiO2/",
            [
                ["Ti5_0.1z_eps_OUTCAR"],
                ["Ti5_0.1x_eps_OUTCAR"],
                [
                    "O43_0.1z_eps_OUTCAR",
                    "O43_m0.1z_eps_OUTCAR",
                ],
                ["O43_0.1x_eps_OUTCAR"],
                ["O43_0.1y_eps_OUTCAR"],
            ],
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_incompatible_model(
    outcar_ref_structure_fixture: ReferenceStructure,
    data_directory: str,
    dof_eps_outcars: list[str],
) -> None:
    """Test a full spectrum calculation using ARTModel."""
    # Setup model
    ref_structure = outcar_ref_structure_fixture
    model = ARTModel(ref_structure, np.zeros((3, 3)))
    for outcar_names in dof_eps_outcars:
        model.add_art_from_files(
            [f"{data_directory}/{name}" for name in outcar_names], file_format="outcar"
        )

    # Spectrum test
    phonons = ramannoodle.io.generic.read_phonons(
        "test/data/STO/phonons_OUTCAR", file_format="outcar"
    )
    with pytest.raises(ValueError) as err:
        phonons.get_raman_spectrum(model)
    assert "polarizability_model and phonons are incompatible" in str(err.value)
