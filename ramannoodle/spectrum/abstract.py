"""Abstract spectra."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class RamanSpectrum(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for Raman spectra."""

    @abstractmethod
    def measure(  # pylint: disable=too-many-arguments
        self,
        orientation: str | NDArray[np.float64] = "polycrystalline",
        laser_correction: bool = False,
        laser_wavelength: float = 522,
        bose_einstein_correction: bool = False,
        temperature: float = 300,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Calculate and return a raw Raman spectrum.

        Parameters
        ----------
        orientation
            Currently only "polycrystalline" is supported. Future versions will support
            arbitrary orientations.
        laser_correction
            Applies laser-wavelength-dependent intensity correction. If True,
            ``laser_wavelength`` must be specified.
        laser_wavelength
            In nm.
        bose_einstein_correction
            Applies temperature-dependent Bose Einstein correction. If True,
            ``temperature`` must be specified.
        temperature
            In Kelvin.

        Returns
        -------
        :
            2-tuple. First element is wavenumbers (cm\ :sup:`-1`), a 1D array. The
            second element is intensities (arbitrary units), a 1D array of the same
            shape.

        """
