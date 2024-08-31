"""Utility functions for spectra."""

import numpy as np
from numpy.typing import NDArray

import scipy
import scipy.fftpack

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
        | (cm\ :sup:`-1`) 1D array with shape (M,).
    intensities
        | (arbitrary units) 1D array with shape (M,).
    function
        | Supports ``"gaussian"`` or ``"lorentzian"``.
    width
        | (cm\ :sup:`-1`)
    out_wavenumbers
        (cm\ :sup:`-1`) 1D array with shape (L,) where L is arbitrary.

        If None, ``out_wavenumbers`` is determined automatically.

    Returns
    -------
    :
        2-tuple:
                0. | wavenumbers (``out_wavenumbers``) --
                   | (cm\ :sup:`-1`) 1D array with shape (L,).
                #. | intensities --
                   | (arbitrary units) 1D array with shape (L,).

    """
    if out_wavenumbers is None:
        min_wavenumber = np.min(wavenumbers) - 100
        max_wavenumber = np.max(wavenumbers) + 100
        num_samples = int(np.rint(max_wavenumber - min_wavenumber))
        out_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, num_samples)
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
                (1 / width)
                * (1 / np.sqrt(2 * np.pi))
                * np.exp(-((wavenumber - out_wavenumbers) ** 2) / (2 * width**2))
            )
        elif function == "lorentzian":
            factor = (1 / np.pi) * (
                0.5 * width / ((wavenumber - out_wavenumbers) ** 2 + (0.5 * width) ** 2)
            )
        else:
            raise ValueError(f"unsupported convolution type: {function}")
        convolved_intensities += factor * intensity
    return (out_wavenumbers, convolved_intensities)


def _calc_autocorrelation(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the positive autocorrelation of a signal.

    Parameters
    ----------
    signal
        Array with shape (S,) where S is the number of samples.

    Returns
    -------
    :
        Array with shape (S,])
    """
    verify_ndarray_shape("signal", signal, (None,))
    autocorrelation = scipy.signal.correlate(signal, signal, "full")
    autocorrelation = autocorrelation[(len(autocorrelation) - 1) // 2 :]
    return autocorrelation


def calc_signal_spectrum(
    signal: NDArray[np.float64], sampling_rate: float
) -> NDArray[np.float64]:
    r"""Calculate a signal's spectrum.

    The spectrum is  defined as the positive-frequency Fourier transform of the
    signal's autocorrelation.

    Parameters
    ----------
    signal
        | Array with shape (S,) where S is the number of samples.
    sampling_rate
        | (fs)

    Returns
    -------
    :
        2-tuple:
            0. | wavenumbers --
               | (cm\ :sup:`-1`) 1D array with shape (ceiling(S / 2),).
            #. | intensities --
               | (arbitrary units) 1D array with shape (ceiling(S / 2),).

    """
    autocorrelation = _calc_autocorrelation(signal)
    wavenumbers = (
        scipy.fftpack.fftfreq(autocorrelation.size, sampling_rate)
        * 33.35640951981521
        * 1e3
    )
    intensities = np.real(scipy.fftpack.fft(autocorrelation))
    return wavenumbers[wavenumbers >= 0], intensities[wavenumbers >= 0]
