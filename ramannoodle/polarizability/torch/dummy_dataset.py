"""Dummy polarizability PyTorch dataset.

Used when torch installation cannot be found.

:meta private:
"""

import numpy as np
from numpy.typing import NDArray

TORCH_PRESENT = False


class PolarizabilityDataset:  # pylint: disable=too-few-public-methods
    """PyTorch dataset of atomic structures and polarizabilities.

    Polarizabilities are scaled and flattened into vectors containing the six
    independent tensor components.

    Parameters
    ----------
    lattices
        | (Å) 3D array with shape (S,3,3) where S is the number of samples.
    atomic_numbers
        | List of length S containing lists of length N, where N is the number of atoms.
    positions
        | (fractional) 3D array with shape (S,N,3).
    polarizabilities
        | 3D array with shape (S,3,3).
    scale_mode
        | Supports ``"standard"`` (standard scaling), ``"stddev"`` (division by
        | standard deviation), and ``"none"`` (no scaling).

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        lattices: NDArray[np.float64],
        atomic_numbers: list[list[int]],
        positions: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        scale_mode: str = "standard",
    ):
        raise ModuleNotFoundError("torch installation not found")
