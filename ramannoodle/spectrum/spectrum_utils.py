"""Utility functions for spectra."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import verify_ndarray_shape, verify_ndarray, get_type_error


def convolve_spectrum(
    wavenumbers: NDArray[np.float64],
    intensities: NDArray[np.float64],
    function: str = "gaussian",
    width: float = 5,
    out_wavenumbers: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Convolve a spectrum, producing a smoothing effect.

    Parameters
    ----------
    wavenumbers
        cm\ :sup:`-1` | 1D array with shape (M,).
    intensities
        Arbitrary units | 1D array with shape (M,).
    function
        Convolution function. Must be either "gaussian" or "lorentzian".
    width
        In cm\ :sup:`-1`.
    out_wavenumbers
        cm\ :sup:`-1` | Optional parameter the output wavenumbers. If None,
        wavenumbers are determined automatically.

    Returns
    -------
    :
        2-tuple containing wavenumbers (cm\ :sup:`-1`) and corresponding intensities
        (arbitrary units).

    """
    if out_wavenumbers is None:
        out_wavenumbers = np.linspace(
            np.min(wavenumbers) - 100, np.max(wavenumbers) + 100, 1000
        )
    verify_ndarray_shape("out_wavenumbers", out_wavenumbers, (None,))
    verify_ndarray_shape("wavenumbers", wavenumbers, (None,))
    verify_ndarray_shape("intensities", intensities, (len(wavenumbers),))
    try:
        if width <= 0:
            raise ValueError(f"invalid width: {width} <= 0")
    except TypeError as exc:
        raise get_type_error("width", width, "float") from exc
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
            raise ValueError(f"unsupported convolution type: {function}")
        convolved_intensities += factor * intensity
    return (out_wavenumbers, convolved_intensities)
