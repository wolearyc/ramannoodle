"""Classes for storing and manipulating spectra."""

import numpy as np
from numpy.typing import NDArray

from . import spectrum_utils


class PhononRamanSpectrum:  # pylint: disable=too-few-public-methods
    """Phonon-based first-order Raman spectrum."""

    def __init__(
        self,
        phonon_wavenumbers: NDArray[np.float64],
        raman_tensors: NDArray[np.float64],
    ) -> None:
        """Construct."""
        self._phonon_wavenumbers = phonon_wavenumbers
        self._raman_tensors = raman_tensors

    def measure(  # pylint: disable=too-many-arguments
        self,
        orientation: str | NDArray[np.float64] = "polycrystalline",
        laser_correction: bool = False,
        laser_wavelength: float | None = None,
        bose_einstein_correction: bool = False,
        temperature: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate and return a largely unprocessed Raman spectrum.

        Parameters
        ----------
        orientation: str | NDArray[np.float64]

        laser_correction:
        """
        if orientation != "polycrystalline":
            raise NotImplementedError(
                "only polycrystalline spectra are supported for now"
            )
        if laser_correction and laser_wavelength is None:
            raise ValueError(
                "laser wavenumber correction requires argument 'laser_wavelength' "
                "to be specified"
            )
        if bose_einstein_correction and temperature is None:
            raise ValueError(
                "bose-einstein correction requires argument 'temperature' "
                "to be specified"
            )

        alpha_squared = (
            (
                self._raman_tensors[:, 0, 0]
                + self._raman_tensors[:, 1, 1]
                + self._raman_tensors[:, 2, 2]
            )
            / 3.0
        ) ** 2
        gamma_squared = (
            (self._raman_tensors[:, 0, 0] - self._raman_tensors[:, 1, 1]) ** 2
            + (self._raman_tensors[:, 0, 0] - self._raman_tensors[:, 2, 2]) ** 2
            + (self._raman_tensors[:, 1, 1] - self._raman_tensors[:, 2, 2]) ** 2
            + 6.0
            * (
                self._raman_tensors[:, 0, 1] ** 2
                + self._raman_tensors[:, 0, 2] ** 2
                + self._raman_tensors[:, 1, 2] ** 2
            )
        ) / 2.0
        intensities = 45.0 * alpha_squared + 7.0 * gamma_squared

        if laser_correction:
            laser_wavenumber = 10000000 / laser_wavelength  # type: ignore
            intensities *= spectrum_utils.get_laser_correction(
                self._phonon_wavenumbers, laser_wavenumber
            )
        if bose_einstein_correction:
            intensities *= spectrum_utils.get_bose_einstein_correction(
                self._phonon_wavenumbers, temperature  # type: ignore
            )

        return self._phonon_wavenumbers, intensities
