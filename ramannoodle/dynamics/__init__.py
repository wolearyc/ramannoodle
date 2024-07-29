"""Classes representing atomic dynamics, including phonons and trajectories."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..spectrum import RamanSpectrum, DensityOfStates, RamanSettings
from ..polarizability import PolarizabilityModel


class Dynamics(ABC):
    """Abstract class for atomic dynamics."""

    @abstractmethod
    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        """Calculate a Raman spectrum."""

    @abstractmethod
    def calculate_density_of_states(self) -> DensityOfStates:
        """Calculate a vibrational density of states."""


class Phonons(Dynamics):
    """Harmonic lattice vibrations.

    A phonon can be represented by a wavenumber and corresponding atomic displacement.
    """

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
        """Calculate a Raman spectrum."""
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Calculate a vibrational density of states."""
        return DensityOfStates(np.array([]), np.array([]))

    def get_wavenumbers(self) -> NDArray[np.float64]:
        """Return wavenumbers in cm-1."""
        return self._wavenumbers

    def get_displacements(self) -> NDArray[np.float64]:
        """Return atomic displacements."""
        return self._displacements


class MDTrajectory(Dynamics):
    """A molecular dynamics trajectory."""

    def __init__(self) -> None:
        pass

    def calculate_raman_spectrum(
        self, polarizability_model: PolarizabilityModel, raman_settings: RamanSettings
    ) -> RamanSpectrum:
        """Calculate a Raman spectrum."""
        return RamanSpectrum(np.array([]), np.array([]))

    def calculate_density_of_states(self) -> DensityOfStates:
        """Calculate a vibrational density of states."""
        return DensityOfStates(np.array([]), np.array([]))
