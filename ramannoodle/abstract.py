"""Abstract classes."""

from abc import ABC, abstractmethod


import numpy as np
from numpy.typing import NDArray


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract polarizability model."""

    @abstractmethod
    def calc_polarizabilities(
        self, positions_batch: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return estimated polarizabilities for a batch of fractional positions.

        Parameters
        ----------
        positions_batch
            (fractional) Array with shape (S,N,3) where S is the number of samples
            and N is the number of atoms.

        Returns
        -------
        :
            Array with shape (S,3,3).
        """


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
            Supports ``"polycrystalline"``. Future versions will support arbitrary
            orientations.
        laser_correction
            If ``True``, applies laser-wavelength-dependent intensity correction.
        laser_wavelength
            (nm) Ignored if ``laser_correction == False``.
        bose_einstein_correction
            If ``True``, applies temperature-dependent Bose Einstein correction.
        temperature
            (K) Ignored if ``bose_einstein_correction == False``.

        Returns
        -------
        :
            0. wavenumbers -- (cm\ :sup:`-1`) Array with shape (M,).

            #. intensities -- (arbitrary units) Array with shape (M,).

        """


class Dynamics(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for atomic dynamics."""

    @abstractmethod
    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> RamanSpectrum:
        """Calculate a raman spectrum using a polarizability model.

        Parameters
        ----------
        polarizability_model
            Must be compatible with the dynamics.
        """
