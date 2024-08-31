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
            Supports ``"polycrystalline"``.

            Future versions will support arbitrary orientations.
        laser_correction
            | Whether to apply laser-wavelength-dependent intensity correction.
        laser_wavelength
            | (nm) Ignored if ``laser_correction == False``.
        bose_einstein_correction
            | Whether to apply temperature-dependent Bose Einstein correction.
        temperature
            | (K) Ignored if ``bose_einstein_correction == False``.

        Returns
        -------
        :
            2-tuple:
                0. | wavenumbers --
                   | (cm\ :sup:`-1`) 1D array with shape (M,).
                #. | intensities --
                   | (arbitrary units) 1D array with shape (M,).

        """
