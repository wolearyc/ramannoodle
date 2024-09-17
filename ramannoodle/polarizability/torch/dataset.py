"""Polarizability PyTorch dataset."""

import copy

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ramannoodle.exceptions import verify_ndarray_shape, verify_list_len, get_type_error
import ramannoodle.polarizability.torch.utils as rn_torch_utils

TORCH_PRESENT = True


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
        verify_ndarray_shape("lattices", lattices, (None, 3, 3))
        num_samples = lattices.shape[0]
        verify_list_len("atomic_numbers", atomic_numbers, num_samples)
        num_atoms = None
        for i, sublist in enumerate(atomic_numbers):
            verify_list_len(f"atomic_numbers[{i}]", sublist, num_atoms)
            if num_atoms is None:
                num_atoms = len(sublist)
        verify_ndarray_shape("positions", positions, (num_samples, num_atoms, 3))
        verify_ndarray_shape(
            "polarizabilities", polarizabilities, (num_samples, None, None)
        )

        default_type = torch.get_default_dtype()
        self._lattices = torch.from_numpy(lattices).type(default_type)
        try:
            self._atomic_numbers = torch.tensor(atomic_numbers).type(torch.int)
        except (TypeError, ValueError) as exc:
            raise get_type_error(
                "atomic_numbers", atomic_numbers, "list[list[int]]"
            ) from exc
        self._positions = torch.from_numpy(positions).type(default_type)
        self._polarizabilities = torch.from_numpy(polarizabilities)

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
    def atomic_numbers(self) -> Tensor:
        """Get (a copy of) atomic numbers.

        Returns
        -------
        :
            2D tensor with size [S,N] where S is the number of samples and N is the
            number of atoms.
        """
        return copy.copy(self._atomic_numbers)

    @property
    def positions(self) -> Tensor:
        """Get (a copy of) positions.

        Returns
        -------
        :
            3D tensor with size [S,N,3] where S is the number of samples and N is the
            number of atoms.
        """
        return self._positions.detach().clone()

    @property
    def polarizabilities(self) -> Tensor:
        """Get (a copy of) polarizabilities.

        Returns
        -------
        :
            3D tensor with size [S,3,3] where S is the number of samples.
        """
        return self._polarizabilities.detach().clone()

    @property
    def scaled_polarizabilities(self) -> Tensor:
        """Get (a copy of) scaled polarizabilities.

        Returns
        -------
        :
            2D tensor with size [S,6] where S is the number of samples.
        """
        return self._scaled_polarizabilities.detach().clone()

    @property
    def mean_polarizability(self) -> Tensor:
        """Get mean polarizability.

        Return
        ------
        :
            2D tensor with size [3,3].
        """
        return self._polarizabilities.mean(0, keepdim=True)

    @property
    def stddev_polarizability(self) -> Tensor:
        """Get standard deviation of polarizability."""
        return self._polarizabilities.std(0, unbiased=False, keepdim=True)

    def scale_polarizabilities(self, mean: Tensor, stddev: Tensor) -> None:
        """Standard-scale polarizabilities given a mean and standard deviation.

        This method may be used to scale validation or test datasets according
        to the mean and standard deviation of the training set, as is best practice.

        Parameters
        ----------
        mean
            | 2D tensor with size [3,3] or 1D tensor.
        stddev
            | 2D tensor with size [3,3] or 1D tensor.

        """
        _, _, scaled = _scale_and_flatten_polarizabilities(
            self._polarizabilities, scale_mode="none"
        )
        try:
            scaled = self._polarizabilities - mean
        except TypeError as exc:
            raise get_type_error("mean", mean, "Tensor") from exc
        except RuntimeError as exc:
            raise rn_torch_utils.get_tensor_size_error(
                "mean", mean, "[3,3] or [1]"
            ) from exc
        try:
            scaled /= stddev
        except TypeError as exc:
            raise get_type_error("stddev", stddev, "Tensor") from exc
        except RuntimeError as exc:
            raise rn_torch_utils.get_tensor_size_error(
                "stddev", stddev, "[3,3] or [1]"
            ) from exc

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
