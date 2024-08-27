"""Abstract polarizability models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract polarizability model."""

    @abstractmethod
    def calc_polarizability(
        self, positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return an estimated polarizability for a set of fractional positions.

        Parameters
        ----------
        positions
            Unitless | 2D array with shape (N,3) where N is the number of atoms.

        Returns
        -------
        :
            Unitless | 2D array with shape (3,3).
        """
