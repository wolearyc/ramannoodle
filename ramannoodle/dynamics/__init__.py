"""Parent class for jiggling atoms."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..spectrum import RamanSpectrum, DensityOfStates, RamanSettings
from ..polarizability import PolarizabilityModel


class Dynamics(ABC):
    """Represents jiggling atoms"""

    @abstractmethod
    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        """Returns a Raman spectrum."""

    @abstractmethod
    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational spectrum."""


class Phonons(Dynamics):
    """A set of phonons, which are represented simply as
    a list of wavenumbers (eigenvectors) and a list of displacements (eigenvalues)"""

    def __init__(
        self, wavenumbers: NDArray[np.float64], displacements: NDArray[np.float64]
    ) -> None:
        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._displacements: NDArray[np.float64] = displacements

    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational spectrum."""
        return DensityOfStates(np.array([]), np.array([]))


class MDTrajectory(Dynamics):
    """A molecular dynamics trajectory."""

    def __init__(self) -> None:
        pass

    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational spectrum."""
        return DensityOfStates(np.array([]), np.array([]))
