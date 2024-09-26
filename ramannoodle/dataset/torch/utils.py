"""Some torch utilities."""

from typing import Sequence

from ramannoodle.exceptions import (
    get_type_error,
    get_torch_missing_error,
)

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError as exc:
    raise get_torch_missing_error() from exc

# pylint complains about torch.norm
# pylint: disable=not-callable


def polarizability_vectors_to_tensors(polarizability_vectors: Tensor) -> Tensor:
    """Convert polarizability vectors to symmetric tensors.

    Parameters
    ----------
    polarizability_vectors
        Tensor with size [S,6].

    Returns
    -------
    :
        Tensor with size [S,3,3].
    """
    verify_tensor_size("polarizability_vectors", polarizability_vectors, (None, 6))
    indices = torch.tensor(
        [
            [0, 3, 4],
            [3, 1, 5],
            [4, 5, 2],
        ]
    )
    return polarizability_vectors[:, indices]


def polarizability_tensors_to_vectors(polarizability_tensors: Tensor) -> Tensor:
    """Convert polarizability tensors to vectors.

    Parameters
    ----------
    polarizability_tensors
        Tensor with size [S,3,3] where S is the number of samples.

    Returns
    -------
    :
        Tensor with size [S,6].

    """
    verify_tensor_size("polarizability_tensors", polarizability_tensors, (None, 3, 3))
    indices = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]).T
    return polarizability_tensors[:, indices[0], indices[1]]


def _get_tensor_size_str(size: Sequence[int | None]) -> str:
    """Get a string representing a tensor size.

    "_" indicates a dimension can be any size.

    Parameters
    ----------
    size
        None indicates dimension can be any size.
    """
    result = "["
    for i in size:
        if i is None:
            result += "_,"
        else:
            result += f"{i},"
    if len(size) == 1:
        return result + "]"
    return result[:-1] + "]"


def get_tensor_size_error(name: str, tensor: Tensor, desired_size: str) -> ValueError:
    """Get ValueError indicating a PyTorch Tensor has the wrong size."""
    try:
        shape_spec = f"{_get_tensor_size_str(tensor.size())} != {desired_size}"
    except AttributeError as exc:
        raise get_type_error("tensor", tensor, "Tensor") from exc
    return ValueError(f"{name} has wrong size: {shape_spec}")


def verify_tensor_size(name: str, tensor: Tensor, size: Sequence[int | None]) -> None:
    """Verify a PyTorch Tensor's size.

    :meta private: We should avoid calling this function whenever possible (EATF).

    Parameters
    ----------
    size
        int elements will be checked, None elements will not be.
    """
    try:
        if len(size) != tensor.ndim:
            raise get_tensor_size_error(name, tensor, _get_tensor_size_str(size))
        for d1, d2 in zip(tensor.size(), size, strict=True):
            if d2 is not None and d1 != d2:
                raise get_tensor_size_error(name, tensor, _get_tensor_size_str(size))
    except AttributeError as exc:
        raise get_type_error(name, tensor, "Tensor") from exc
