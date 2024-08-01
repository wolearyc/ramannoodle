"""Classes representing atomic dynamics, including phonons and trajectories."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..spectrum import PhononRamanSpectrum
from ..polarizability import PolarizabilityModel
from ..globals import RAMAN_TENSOR_CENTRAL_DIFFERENCE


class Dynamics(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for atomic dynamics."""

    @abstractmethod
    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a Raman spectrum."""


class Phonons(Dynamics):
    """Harmonic lattice vibrations.

    A phonon can be represented by a wavenumber and corresponding atomic displacement.
    The wavenumbers are the eigenvalues of the system's dynamical matrix, while the
    atomic displacements are the eigenvectors of the dynamical matrix divided by the
    square root of the atomic masses.

    Parameters
    ----------
    wavenumbers
        1D array with length M
    cartesian_displacements
        3D array with shape (M,N,3) where N is the number of atoms

    """

    def __init__(
        self,
        wavenumbers: NDArray[np.float64],
        cartesian_displacements: NDArray[np.float64],
    ) -> None:
        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._cartesian_displacements: NDArray[np.float64] = cartesian_displacements

    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a Raman spectrum."""
        raman_tensors = []
        for cartesian_displacement in self._cartesian_displacements:
            plus = polarizability_model.get_polarizability(
                cartesian_displacement * RAMAN_TENSOR_CENTRAL_DIFFERENCE
            )
            minus = polarizability_model.get_polarizability(
                -cartesian_displacement * RAMAN_TENSOR_CENTRAL_DIFFERENCE
            )
            raman_tensors.append((plus - minus) / RAMAN_TENSOR_CENTRAL_DIFFERENCE)

        return PhononRamanSpectrum(self._wavenumbers, np.array(raman_tensors))

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Return wavenumbers.

        Returns
        -------
        :
            1D array with length M
        """
        return self._wavenumbers

    def get_cartesian_displacements(self) -> NDArray[np.float64]:
        """Return cartesian displacements.

        Returns
        -------
        :
            3D array with shape (M,N,3) where M is the number of displacements
            and N is the number of atoms
        """
        return self._cartesian_displacements
