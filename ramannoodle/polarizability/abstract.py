"""Abstract polarizability models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract polarizability model."""

    @abstractmethod
    def calc_polarizabilities(
        self, positions_batch: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return estimated polarizabilities for a batch of fractional positions.

        Parameters
        ----------
        positions_batch
            | (fractional) 3D array with shape (S,N,3) where S is the number of samples
            | and N is the number of atoms.

        Returns
        -------
        :
            3D array with shape (S,3,3).
        """
