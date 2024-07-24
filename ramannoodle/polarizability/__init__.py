"""Parent class for polarizability models."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Represents a polarizability model"""

    @abstractmethod
    def get_polarizability(
        self, displacements: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Returns a polarizability for a set of atomic displacements."""