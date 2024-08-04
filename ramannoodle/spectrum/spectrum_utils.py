"""Utility functions for spectra."""

import numpy as np
from numpy.typing import NDArray

from ..globals import BOLTZMANN_CONSTANT, verify_ndarray_shape, verify_ndarray


def get_bose_einstein_correction(
    wavenumbers: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """Calculate Bose-Einstein spectral correction.

    Parameters
    ----------
    wavenumbers
    temperature
        in kelvin

    Returns
    -------
    :
        Correction factor for each wavenumber.

    Raises
    ------
    ValueError

    """
    if temperature <= 0:
        raise ValueError(f"invalid temperature: {temperature}<=0")
    try:
        energy = wavenumbers * 29979245800 * 4.1357e-15  # in eV
        return 1 / (1 - np.exp(-energy / (BOLTZMANN_CONSTANT * temperature)))
    except TypeError as exc:
        wrong_type = type(wavenumbers).__name__
        raise TypeError(
            f"wavenumbers should be an ndarray, not a {wrong_type}"
        ) from exc


def get_laser_correction(
    wavenumbers: NDArray[np.float64], laser_wavenumber: float
) -> NDArray[np.float64]:
    """Calculate conventional laser-wavenumber-dependent spectral correction.

    Parameters
    ----------
    wavenumbers
    laser_wavenumber

    Returns
    -------
    :
        Correction factor for each wavenumber.

    Raises
    ------
    ValueError

    """
    if laser_wavenumber <= 0:
        raise ValueError(f"invalid laser wavenumber: {laser_wavenumber}<=0")
    try:
        return ((wavenumbers - laser_wavenumber) / 10000) ** 4 / wavenumbers
    except TypeError as exc:
        wrong_type = type(wavenumbers).__name__
        raise TypeError(
            f"wavenumbers should be an ndarray, not a {wrong_type}"
        ) from exc


def convolve_intensities(
    wavenumbers: NDArray[np.float64],
    intensities: NDArray[np.float64],
    function: str = "gaussian",
    width: float = 5,
    out_wavenumbers: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convolve a spectrum, producing a smoothing effect.

    Parameters
    ----------
    wavenumbers
        input wavenumbers
    intensities
        input intensities
    function
        convolution function. must be either "gaussian" or "lorentzian"
    width
    out_wavenumbers
        Optional parameter the output wavenumbers. If None, wavenumbers are
        determined automatically.

    Returns
    -------
    :
        2-tuple containing wavenumbers and corresponding intensities

    Raises
    ------
    ValueError
    TypeError
    """
    if out_wavenumbers is None:
        out_wavenumbers = np.linspace(
            np.min(wavenumbers) - 100, np.max(wavenumbers) + 100, 1000
        )
    out_wavenumbers = np.array(out_wavenumbers)  # to shut the type checker up
    verify_ndarray_shape("wavenumbers", wavenumbers, (None,))
    verify_ndarray_shape("intensities", intensities, (len(wavenumbers),))
    if width <= 0:
        raise ValueError(f"invalid width: {width} <= 0")
    verify_ndarray("out_wavenumbers", out_wavenumbers)

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
