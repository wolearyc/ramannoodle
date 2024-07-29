"""Classes for storing and manipulating spectra."""

import numpy as np
from numpy.typing import NDArray


class DensityOfStates:  # pylint: disable=too-few-public-methods
    """Vibrational spectrum."""

    def __init__(
        self, raw_wavenumbers: NDArray[np.float64], raw_intensities: NDArray[np.float64]
    ) -> None:
        self._raw_wavenumbers = raw_wavenumbers
        self._raw_intensities = raw_intensities

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Return wavenumbers."""
        return self._raw_wavenumbers

    def get_intensities(self) -> NDArray[np.float64]:
        """Return intensities."""
        return self._raw_intensities


class RamanSpectrum:
    """Raman spectrum."""

    def __init__(
        self, raw_wavenumbers: NDArray[np.float64], raw_intensities: NDArray[np.float64]
    ) -> None:
        self._raw_wavenumbers = raw_wavenumbers
        self._raw_intensities = raw_intensities

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Return wavenumbers."""
        return self._raw_wavenumbers

    def get_intensities(self) -> NDArray[np.float64]:
        """Return intensities."""
        return self._raw_intensities


class RamanSettings:  # pylint: disable=too-few-public-methods
    """Settings for a Raman calculation.

    This class currently has no function. In the future, it will specify the parameters
    of a virtual Raman experiment, such as laser wavelength, measurement angle, and
    sample orientation.

    """

    def __init__(self) -> None:
        self._polycrystalline = True
        # self._angle = angle
