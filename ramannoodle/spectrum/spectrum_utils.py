"""Utility functions for spectra."""

import numpy as np
from numpy.typing import NDArray

from ..globals import BOLTZMANN_CONSTANT


def get_bose_einstein_correction(
    wavenumbers: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """Calculate Bose-Einstein spectral correction.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
    T : float
        Temperature in K

    Returns
    -------
    numpy.ndarray
        Correction factor for each wavenumber.

    """
    energy = wavenumbers * 29979245800 * 4.1357e-15  # in eV
    return 1 / (1 - np.exp(-energy / (BOLTZMANN_CONSTANT * temperature)))


def get_laser_correction(
    wavenumbers: NDArray[np.float64], laser_wavenumber: float
) -> NDArray[np.float64]:
    """Calculate conventional laser-wavenumber-dependent spectral correction.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
    laser_wavenumber: float

    Returns
    -------
    numpy.ndarray
        Correction factor for each wavenumber.

    """
    return ((wavenumbers - laser_wavenumber) / 10000) ** 4 / wavenumbers


def convolve_intensities(
    wavenumbers: NDArray[np.float64],
    intensities: NDArray[np.float64],
    function: str = "gaussian",
    width: float = 5,
    out_wavenumbers: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convolves and smooths intensities."""
    if out_wavenumbers is None:
        out_wavenumbers = np.linspace(np.min(wavenumbers), np.max(wavenumbers), 1000)

    out_wavenumbers = np.array(out_wavenumbers)
    convolved_intensities = out_wavenumbers * 0
    for wavenumber, intensity in zip(wavenumbers, intensities):
        factor = 0
        if function == "gaussian":
            factor = (
                1
                / width
                * 1
                / np.sqrt(2 * np.pi)
                * np.exp(-((wavenumber - out_wavenumbers) ** 2) / (2 * width**2))
            )
        elif function == "lorentzian":
            factor = (
                1
                / np.pi
                * 0.5
                * width
                / ((wavenumber - out_wavenumbers) ** 2 + (0.5 * width) ** 2)
            )
        else:
            raise ValueError(f"unsupported convolution type: {type}")
        convolved_intensities += factor * intensity
    return (out_wavenumbers, convolved_intensities)
