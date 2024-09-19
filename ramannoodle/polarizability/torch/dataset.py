"""Polarizability PyTorch dataset."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (
    verify_ndarray_shape,
    verify_list_len,
    get_torch_missing_error,
)

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    import ramannoodle.polarizability.torch.utils as rn_torch_utils
except ModuleNotFoundError as exc:
    raise get_torch_missing_error() from exc


def _scale_and_flatten_polarizabilities(
    polarizabilities: Tensor,
    scale_mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale and flatten polarizabilities.

    3x3 polarizabilities are flattened into 6-vectors: (xx,yy,zz,xy,xz,yz).

    Parameters
    ----------
    polarizabilities
        | 3D tensor with size [S,3,3] where S is the number of samples.
    scale_mode
        | Supports ``"standard"`` (standard scaling), ``"stddev"`` (division by
        | standard deviation), and ``"none"`` (no scaling).

    Returns
    -------
    :
        3-tuple:
            0. | mean --
               | Element-wise mean of polarizabilities.
            #. | standard deviation --
               | Element-wise standard deviation of polarizabilities.
            #. | polarizability vectors --
               | 2D tensor with size [S,6].

    """
    rn_torch_utils.verify_tensor_size(
        "polarizabilities", polarizabilities, [None, 3, 3]
    )

    mean = polarizabilities.mean(0, keepdim=True)
    stddev = polarizabilities.std(0, unbiased=False, keepdim=True)
    if scale_mode == "standard":
        polarizabilities = (polarizabilities - mean) / stddev
    elif scale_mode == "stddev":
        polarizabilities = (polarizabilities - mean) / stddev + mean
    elif scale_mode != "none":
        raise ValueError(f"unsupported scale mode: {scale_mode}")

    scaled_polarizabilities = torch.zeros((polarizabilities.size(0), 6))
    scaled_polarizabilities[:, 0] = polarizabilities[:, 0, 0]
    scaled_polarizabilities[:, 1] = polarizabilities[:, 1, 1]
    scaled_polarizabilities[:, 2] = polarizabilities[:, 2, 2]
    scaled_polarizabilities[:, 3] = polarizabilities[:, 0, 1]
    scaled_polarizabilities[:, 4] = polarizabilities[:, 0, 2]
    scaled_polarizabilities[:, 5] = polarizabilities[:, 1, 2]

    return mean, stddev, scaled_polarizabilities


class PolarizabilityDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """PyTorch dataset of atomic structures and polarizabilities.

    Polarizabilities are scaled and flattened into 6-vectors containing the
    independent tensor components.

    Parameters
    ----------
    lattice
        | (Å) Array with shape (3,3).
    atomic_numbers
        | List of length N where N is the number of atoms.
    positions
        | (fractional) 3D array with shape (S,N,3) where S is the number of samples.
    polarizabilities
        | 3D array with shape (S,3,3).
    scale_mode
        | Supports ``"standard"`` (standard scaling), ``"stddev"`` (division by
        | standard deviation), and ``"none"`` (no scaling).

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        lattice: NDArray[np.float64],
        atomic_numbers: list[int],
        positions: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        scale_mode: str = "standard",
    ):
        # Validate parameter shapes
        verify_ndarray_shape("lattice", lattice, (3, 3))
        verify_list_len("atomic_numbers", atomic_numbers, None)
        num_atoms = len(atomic_numbers)
        verify_ndarray_shape("positions", positions, (None, num_atoms, 3))
        num_samples = positions.shape[0]
        verify_ndarray_shape("polarizabilities", polarizabilities, (num_samples, 3, 3))

        default_type = torch.get_default_dtype()
        self._lattices = torch.tensor(lattice).type(default_type).unsqueeze(0)
        self._lattices = self._lattices.expand(num_samples, 3, 3)
        self._atomic_numbers = torch.tensor(atomic_numbers).type(torch.int).unsqueeze(0)
        self._atomic_numbers = self._atomic_numbers.expand(num_samples, num_atoms)
        self._positions = torch.tensor(positions).type(default_type)
        self._polarizabilities = torch.tensor(polarizabilities)

        _, _, scaled = _scale_and_flatten_polarizabilities(
            self._polarizabilities, scale_mode=scale_mode
        )
        self._scaled_polarizabilities = scaled.type(default_type)

    @property
    def num_atoms(self) -> int:
        """Get number of atoms per sample."""
        return self._positions.size(1)

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return self._positions.size(0)

    @property
    def atomic_numbers(self) -> list[int]:
        """Get (a copy of) atomic numbers.

        Returns
        -------
        :
            List of length N where N is the number of atoms.
        """
        return [int(n) for n in self._atomic_numbers[0]]

    @property
    def positions(self) -> NDArray[np.float64]:
        """Get (a copy of) positions.

        Returns
        -------
        :
            Array with shape (S,N,3) where S is the number of samples and N is the
            number of atoms.
        """
        return self._positions.detach().clone().numpy()

    @property
    def polarizabilities(self) -> NDArray[np.float64]:
        """Get (a copy of) polarizabilities.

        Returns
        -------
        :
            3D array with shape (S,3,3) where S is the number of samples.
        """
        return self._polarizabilities.detach().clone().numpy()

    @property
    def scaled_polarizabilities(self) -> NDArray[np.float64]:
        """Get (a copy of) scaled polarizabilities.

        Returns
        -------
        :
            2D array with shape (S,6) where S is the number of samples.
        """
        return self._scaled_polarizabilities.detach().clone().numpy()

    @property
    def mean_polarizability(self) -> NDArray[np.float64]:
        """Get mean polarizability.

        Return
        ------
        :
            2D array with shape (3,3).
        """
        return self._polarizabilities.mean(0, keepdim=True).clone().numpy()

    @property
    def stddev_polarizability(self) -> NDArray[np.float64]:
        """Get standard deviation of polarizabilities.

        Return
        ------
        :
            2D array with shape (3,3).
        """
        result = self._polarizabilities.std(0, unbiased=False, keepdim=True)
        return result.clone().numpy()

    def scale_polarizabilities(
        self, mean: NDArray[np.float64], stddev: NDArray[np.float64]
    ) -> None:
        """Standard-scale polarizabilities given a mean and standard deviation.

        This method may be used to scale validation or test datasets according
        to the mean and standard deviation of the training set, as is best practice.

        Parameters
        ----------
        mean
            | Array with shape (3,3).
        stddev
            | Array with shape (3,3).

        """
        verify_ndarray_shape("mean", mean, (3, 3))
        verify_ndarray_shape("mean", stddev, (3, 3))

        _, _, scaled = _scale_and_flatten_polarizabilities(
            self._polarizabilities, scale_mode="none"
        )
        scaled = self._polarizabilities - torch.tensor(mean)
        scaled /= stddev
        self._scaled_polarizabilities = scaled

    def __len__(self) -> int:
        """Get number of samples."""
        return self.num_samples

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get lattice, atomic numbers, positions, and scaled polarizabilities."""
        return (
            self._lattices[i],
            self._atomic_numbers[i],
            self._positions[i],
            self._scaled_polarizabilities[i],
        )
