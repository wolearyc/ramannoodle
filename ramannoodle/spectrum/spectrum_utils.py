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
