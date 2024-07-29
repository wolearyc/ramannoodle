"""Spectra."""

import numpy as np
from numpy.typing import NDArray


class DensityOfStates:  # pylint: disable=too-few-public-methods
    """Vibrational spectrum, with useful post-processing methods"""

    def __init__(
        self, raw_wavenumbers: NDArray[np.float64], raw_intensities: NDArray[np.float64]
    ) -> None:
        self._raw_wavenumbers = raw_wavenumbers
        self._raw_intensities = raw_intensities

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Returns wavenumbers"""
        return self._raw_wavenumbers

    def get_intensities(self) -> NDArray[np.float64]:
        """Returns intensities"""
        return self._raw_intensities


class RamanSpectrum:  # pylint: disable=too-few-public-methods
    """Raman spectrum, with useful post-processing methods"""

    def __init__(
        self, raw_wavenumbers: NDArray[np.float64], raw_intensities: NDArray[np.float64]
    ) -> None:
        self._raw_wavenumbers = raw_wavenumbers
        self._raw_intensities = raw_intensities

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Returns wavenumbers"""
        return self._raw_wavenumbers

    def get_intensities(self) -> NDArray[np.float64]:
        """Returns intensities"""
        return self._raw_intensities


class RamanSettings:  # pylint: disable=too-few-public-methods
    """Settings for a Raman calculation. Currently, only
    polycrystalline spectra are calculated."""

    def __init__(self) -> None:
        self._polycrystalline = True
        # self._angle = angle
