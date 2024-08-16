"""Abstract atomic motions."""

from abc import ABC, abstractmethod

from ramannoodle.spectrum.raman import PhononRamanSpectrum
from ramannoodle.polarizability.abstract import PolarizabilityModel


class Dynamics(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for atomic dynamics."""

    @abstractmethod
    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a raman spectrum using a polarizability model.

        Parameters
        ----------
        polarizability_model
            must be compatible with the dynamics
        """
