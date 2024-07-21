"""Parent class for jiggling atoms."""

from abc import ABC, abstractmethod

from ..spectrum.raman import RamanSpectrum
from ..spectrum.vibrational import VibrationalSpectrum


class Dynamics(ABC):
    """Represents jiggling atoms"""

    @abstractmethod
    def calculate_raman_spectrum(self) -> RamanSpectrum:
        """Returns a Raman spectrum."""

    @abstractmethod
    def calculate_vibrational_spectrum(self) -> VibrationalSpectrum:
        """Returns a vibrational spectrum."""
