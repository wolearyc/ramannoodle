"""Abstract atomic motions."""

from abc import ABC, abstractmethod

from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.spectrum.abstract import RamanSpectrum


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
            | Must be compatible with the dynamics.
        """
