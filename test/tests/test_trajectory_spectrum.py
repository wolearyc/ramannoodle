"""Testing for spectra."""

import numpy as np
from numpy.typing import NDArray
import pytest

from ramannoodle.spectrum.spectrum_utils import calc_signal_spectrum
import ramannoodle.io.vasp as vasp_io
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.polarizability.interpolation import InterpolationModel

# pylint: disable=R0801


@pytest.mark.parametrize(
    "signal, sampling_rate",
    [
        (np.random.random(40), 1.0),
        (np.random.random(51), 1.0),
    ],
)
def test_calc_signal_spectrum(
    signal: NDArray[np.float64],
    sampling_rate: float,
) -> None:
    """Test calc_signal_spectrum."""
    wavenumbers, intensities = calc_signal_spectrum(signal, sampling_rate)
    assert wavenumbers.shape == (int(np.ceil(len(signal) / 2)),)
    assert intensities.shape == wavenumbers.shape


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,data_directory,dof_eps_outcars",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            "test/data/TiO2",
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
    indirect=["outcar_ref_structure_fixture"],
)
def test_spectrum(
    outcar_ref_structure_fixture: ReferenceStructure,
    data_directory: str,
    dof_eps_outcars: list[str],
) -> None:
    """Test a full spectrum calculation using InterpolationModel."""
    # Setup model
    ref_structure = outcar_ref_structure_fixture
    _, ref_polarizability = vasp_io.outcar.read_positions_and_polarizability(
        f"{data_directory}/ref_eps_OUTCAR"
    )
    model = InterpolationModel(ref_structure, ref_polarizability)
    for outcar_names in dof_eps_outcars:
        model.add_dof_from_files(
            [f"{data_directory}/{name}" for name in outcar_names],
            file_format="outcar",
            interpolation_order=2,
        )

    # Spectrum test
    with np.load(f"{data_directory}/known_md_spectrum.npz") as known_spectrum:
        trajectory = vasp_io.xdatcar.read_trajectory(
            f"{data_directory}/MD_trajectory_XDATCAR", timestep=5
        )
        spectrum = trajectory.get_raman_spectrum(model)
        wavenumbers, intensities = spectrum.measure(
            laser_correction=True,
            laser_wavelength=532,
            bose_einstein_correction=True,
            temperature=300,
        )

        known_wavenumbers = known_spectrum["wavenumbers"]
        known_intensities = known_spectrum["intensities"]

        assert np.isclose(wavenumbers, known_wavenumbers).all()
        assert np.isclose(intensities, known_intensities, atol=1e-3).all()
