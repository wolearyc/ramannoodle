"""Testing for spectra."""

from pathlib import Path
from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.polarizability.interpolation import InterpolationPolarizabilityModel
from ramannoodle.symmetry import StructuralSymmetry
from ramannoodle import io
from ramannoodle.spectrum.spectrum_utils import (
    convolve_intensities,
    get_bose_einstein_correction,
    get_laser_correction,
)

# pylint: disable=protected-access


def _get_all_eps_outcars(directory: str) -> list[str]:
    """Return name of all *eps_OUTCARs in a directory."""
    path = Path(directory)
    return [str(item) for item in path.glob("*eps_OUTCAR")]


def _validate_polarizabilities(
    model: InterpolationPolarizabilityModel, data_directory: str
) -> None:
    """Check that a model accurately predicts polarizabilities.

    This function will use all *eps_OUTCAR's in a directory as references.
    """
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


@pytest.mark.parametrize(
    "spectrum_path,known_gaussian_spectrum_path,known_lorentzian_spectrum_path",
    [
        (
            "test/data/TiO2/known_spectrum.npz",
            "test/data/TiO2/known_gaussian_spectrum.npz",
            "test/data/TiO2/known_lorentzian_spectrum.npz",
        )
    ],
)
def test_convolve_intensities(
    spectrum_path: str,
    known_gaussian_spectrum_path: str,
    known_lorentzian_spectrum_path: str,
) -> None:
    """Test convolutions."""
    with np.load(spectrum_path) as spectrum:
        wavenumbers = spectrum["wavenumbers"]
        intensities = spectrum["intensities"]

        gaussian_wavenumbers, gaussian_intensities = convolve_intensities(
            wavenumbers, intensities, "gaussian"
        )
        with np.load(known_gaussian_spectrum_path) as known_spectrum:
            known_wavenumbers = known_spectrum["wavenumbers"]
            known_intensities = known_spectrum["intensities"]
            assert np.isclose(gaussian_wavenumbers, known_wavenumbers).all()
            assert np.isclose(gaussian_intensities, known_intensities).all()

        lorentzian_wavenumbers, lorentzian_intensities = convolve_intensities(
            wavenumbers, intensities, "lorentzian"
        )
        with np.load(known_lorentzian_spectrum_path) as known_spectrum:
            known_wavenumbers = known_spectrum["wavenumbers"]
            known_intensities = known_spectrum["intensities"]
            assert np.isclose(lorentzian_wavenumbers, known_wavenumbers).all()
            assert np.isclose(lorentzian_intensities, known_intensities).all()


@pytest.mark.parametrize(
    "wavenumbers,intensities,function,width,out_wavenumbers,exception_type,in_reason",
    [
        (
            np.array([1, 2, 3]),
            [0, 3, 0],
            "gaussian",
            5,
            None,
            TypeError,
            "intensities should have type ndarray, not list",
        ),
        (
            [1, 2, 3],
            np.array([0, 3, 0]),
            "gaussian",
            5,
            None,
            TypeError,
            "wavenumbers should have type ndarray, not list",
        ),
        (
            np.array([1, 2, 3, 4]),
            np.array([0, 3, 0]),
            "gaussian",
            5,
            None,
            ValueError,
            "intensities has wrong shape: (3,) != (4,)",
        ),
        (
            np.array([[1, 2, 3, 5], [2, 3, 4, 5]]),
            np.array([0, 3, 0, 4]),
            "gaussian",
            5,
            None,
            ValueError,
            "wavenumbers has wrong shape: (2,4) != (_,)",
        ),
        (
            np.array([1, 2, 3]),
            np.array([0, 3, 0]),
            "smoother",
            5,
            None,
            ValueError,
            "unsupported convolution type: smoother",
        ),
        (
            np.array([1, 2, 3]),
            np.array([0, 3, 0]),
            "gaussian",
            -4,
            None,
            ValueError,
            "invalid width: -4 <= 0",
        ),
        (
            np.array([1, 2, 3]),
            np.array([0, 3, 0]),
            "gaussian",
            "not_a_int",
            None,
            TypeError,
            "width should have type float, not str",
        ),
        (
            np.array([1, 2, 3]),
            np.array([0, 3, 0]),
            "gaussian",
            5.0,
            np.array(([1, 2], [1, 2])),
            ValueError,
            "out_wavenumbers has wrong shape: (2,2) != (_,)",
        ),
    ],
)
def test_convolve_intensities_exception(  # pylint: disable=too-many-arguments
    wavenumbers: NDArray[np.float64],
    intensities: NDArray[np.float64],
    function: str,
    width: float,
    out_wavenumbers: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test exceptions raised by convolve_intensities."""
    with pytest.raises(exception_type) as error:
        convolve_intensities(wavenumbers, intensities, function, width, out_wavenumbers)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "wavenumbers,temperature,exception_type,in_reason",
    [
        (
            [1, 2, 3],
            300,
            TypeError,
            "wavenumbers should have type ndarray, not list",
        ),
        (
            [1, 2, 3],
            -300,
            ValueError,
            "invalid temperature: -300 <= 0",
        ),
        (
            [1, 2, 3],
            "string temperature",
            TypeError,
            "temperature should have type float, not str",
        ),
    ],
)
def test_get_bose_einstein_correction_exceptions(
    wavenumbers: NDArray[np.float64],
    temperature: float,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test exceptions raised by get_bose_einstein_correction."""
    with pytest.raises(exception_type) as error:
        get_bose_einstein_correction(wavenumbers, temperature)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "wavenumbers,laser_wavenumber,exception_type,in_reason",
    [
        (
            [1, 2, 3],
            600,
            TypeError,
            "wavenumbers should have type ndarray, not list",
        ),
        (
            [1, 2, 3],
            -600,
            ValueError,
            "invalid laser_wavenumber: -600 <= 0",
        ),
        (
            [1, 2, 3],
            "string wavelength",
            TypeError,
            "laser_wavenumber should have type float, not str",
        ),
    ],
)
def test_get_laser_correction(
    wavenumbers: NDArray[np.float64],
    laser_wavenumber: float,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test exceptions raised by get_laser_correction."""
    with pytest.raises(exception_type) as error:
        get_laser_correction(wavenumbers, laser_wavenumber)
    assert in_reason in str(error.value)
