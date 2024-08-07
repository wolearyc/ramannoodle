"""Classes for various polarizability models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract polarizability model."""

    @abstractmethod
    def get_polarizability(
        self, cartesian_displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return an estimated polarizability for a given cartesian displacement.

        Parameters
        ----------
        cartesian_displacement
            2D array with shape (N,3) where N is the number of atoms

        Returns
        -------
        :
            2D array with shape (3,3)
        """
