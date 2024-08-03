"""Testing for spectra."""

from pathlib import Path

import numpy as np

import pytest

from ramannoodle.polarizability.interpolation import InterpolationPolarizabilityModel
from ramannoodle.symmetry import StructuralSymmetry
from ramannoodle import io

# pylint: disable=protected-access


def _get_all_eps_outcars(directory: str) -> list[str]:
    """Return name of all eps OUTCARs in a directory."""
    path = Path(directory)
    return [str(item) for item in path.glob("*eps_OUTCAR")]


def _validate_polarizabilities(
    model: InterpolationPolarizabilityModel, data_directory: str
) -> None:
    for outcar_path in _get_all_eps_outcars(data_directory):
        positions, known_polarizability = io.read_positions_and_polarizability(
            f"{outcar_path}", file_format="outcar"
        )
        cartesian_displacement = model._structural_symmetry.get_cartesian_displacement(
            positions - model._structural_symmetry.get_fractional_positions()
        )
        model_polarizability = model.get_polarizability(cartesian_displacement)
        assert np.isclose(model_polarizability, known_polarizability, atol=1e-4).all()


@pytest.mark.parametrize(
    "outcar_symmetry_fixture,data_directory,dof_eps_outcars",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            "test/data/TiO2/",
            [
                ["Ti5_0.1z_eps_OUTCAR", "Ti5_0.2z_eps_OUTCAR"],
                ["Ti5_0.1x_eps_OUTCAR", "Ti5_0.2x_eps_OUTCAR"],
                [
                    "O43_0.1z_eps_OUTCAR",
                    "O43_0.2z_eps_OUTCAR",
                    "O43_m0.1z_eps_OUTCAR",
                    "O43_m0.2z_eps_OUTCAR",
                ],
                ["O43_0.1x_eps_OUTCAR", "O43_0.2x_eps_OUTCAR"],
                ["O43_0.1y_eps_OUTCAR", "O43_0.2y_eps_OUTCAR"],
            ],
        ),
    ],
    indirect=["outcar_symmetry_fixture"],
)
def test_spectrum(
    outcar_symmetry_fixture: StructuralSymmetry,
    data_directory: str,
    dof_eps_outcars: list[str],
) -> None:
    """Test a spectrum calculation."""
    # Setup model
    symmetry = outcar_symmetry_fixture
    _, polarizability = io.read_positions_and_polarizability(
        f"{data_directory}/ref_eps_OUTCAR", file_format="outcar"
    )
    model = InterpolationPolarizabilityModel(symmetry, polarizability)
    for outcar_names in dof_eps_outcars:
        model.add_dof_from_files(
            [f"{data_directory}/{name}" for name in outcar_names],
            file_format="outcar",
            interpolation_order=2,
        )

    _validate_polarizabilities(model, data_directory)

    # Spectrum test
    with np.load(f"{data_directory}/known_spectrum.npz") as known_spectrum:
        phonons = io.read_phonons(
            f"{data_directory}/phonons_OUTCAR", file_format="outcar"
        )
        spectrum = phonons.get_raman_spectrum(model)
        wavenumbers, intensities = spectrum.measure(
            laser_correction=True,
            laser_wavelength=532,
            bose_einstein_correction=True,
            temperature=300,
        )

        known_wavenumbers = known_spectrum["wavenumbers"]
        known_intensities = known_spectrum["intensities"]

        assert np.isclose(wavenumbers, known_wavenumbers).all()
        assert np.isclose(intensities, known_intensities, atol=1e-4).all()
