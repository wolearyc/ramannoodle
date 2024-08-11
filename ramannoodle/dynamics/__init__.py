"""Classes representing atomic dynamics, including phonons and trajectories."""

from abc import ABC, abstractmethod

from ramannoodle.spectrum.raman import PhononRamanSpectrum
from ramannoodle.polarizability import PolarizabilityModel


class Dynamics(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for atomic dynamics."""

    @abstractmethod
    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a Raman spectrum."""
