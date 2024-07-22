"""Handles the motion of atoms. """

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..spectrum import RamanSpectrum, DensityOfStates, RamanSettings
from ..polarizability import PolarizabilityModel


class Dynamics(ABC):
    """A set of moving atoms."""

    @abstractmethod
    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        """Returns a Raman spectrum."""

    @abstractmethod
    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational density of states."""


class Phonons(Dynamics):
    """Phonons, which can be thought of as a list of wavenumbers (eigenvalues)
    and corresponding atomic displacements (eigenvectors)."""

    def __init__(
        self,
        wavenumbers: NDArray[np.float64],
        displacements: NDArray[np.float64],
    ) -> None:
        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._displacements: NDArray[np.float64] = displacements

    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational density of states."""
        return DensityOfStates(np.array([]), np.array([]))

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Returns wavenumbers in cm-1"""
        return self._wavenumbers

    def get_displacements(self) -> NDArray[np.float64]:
        """Returns displacements"""
        return self._displacements


class MDTrajectory(Dynamics):
    """A molecular dynamics trajectory."""

    def __init__(self) -> None:
        pass

    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Returns a vibrational density of states."""
        return DensityOfStates(np.array([]), np.array([]))
