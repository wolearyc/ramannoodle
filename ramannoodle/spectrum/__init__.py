"""Classes for storing and manipulating spectra."""

import numpy as np
from numpy.typing import NDArray

from . import spectrum_utils
from ..exceptions import verify_ndarray_shape


class PhononRamanSpectrum:  # pylint: disable=too-few-public-methods
    """Phonon-based first-order Raman spectrum.

    Parameters
    ----------
    phonon_wavenumbers
        an ndarray with shape (M,N,3)
    raman_tensors
        an ndarray with shape (M,3,3)

    """

    def __init__(
        self,
        phonon_wavenumbers: NDArray[np.float64],
        raman_tensors: NDArray[np.float64],
    ) -> None:
        verify_ndarray_shape("phonon_wavenumbers", phonon_wavenumbers, (None,))
        verify_ndarray_shape(
            "raman_tensors", raman_tensors, (len(phonon_wavenumbers), 3, 3)
        )
        self._phonon_wavenumbers = phonon_wavenumbers
        self._raman_tensors = raman_tensors

    def measure(  # pylint: disable=too-many-arguments
        self,
        orientation: str | NDArray[np.float64] = "polycrystalline",
        laser_correction: bool = False,
        laser_wavelength: float = 522,
        bose_einstein_correction: bool = False,
        temperature: float = 300,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate and return a raw Raman spectrum.

        Parameters
        ----------
        orientation
            Currently only "polycrystalline" is supported.
        laser_correction
            Applies laser-wavelength-dependent intensity correction. If True,
            `laser_wavelength` must be specified.
        laser_wavelength
        bose_einstein_correction
            Applies temperature-dependent Bose Einstein correction. If True,
            `temperature` must be specified.

        Returns
        -------
        :
            2-tuple, whose first element is wavenumbers, a 1D array with length M
            where M is the number of normal modes. The second element is intensities,
            a 1D array with length M.

        Raises
        ------
        NotImplementedError
            Raised when any orientation besides "polycrystalline" is supplied.

        """
        if orientation != "polycrystalline":
            raise NotImplementedError(
                "only polycrystalline spectra are supported for now"
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
            laser_wavenumber = 10000000 / laser_wavelength
            intensities *= spectrum_utils.get_laser_correction(
                self._phonon_wavenumbers, laser_wavenumber
            )
        if bose_einstein_correction:
            intensities *= spectrum_utils.get_bose_einstein_correction(
                self._phonon_wavenumbers, temperature
            )

        return self._phonon_wavenumbers, intensities
