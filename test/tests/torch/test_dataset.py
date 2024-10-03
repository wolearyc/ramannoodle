"""Testing for PyTorch dataset."""

from typing import Type
import re

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.generic as generic_io
from ramannoodle.dataset.torch._dataset import PolarizabilityDataset


@pytest.mark.parametrize(
    "filepaths, file_format",
    [
        (
            [
                "test/data/TiO2/O43_0.1x_eps_OUTCAR",
                "test/data/TiO2/O43_0.1y_eps_OUTCAR",
                "test/data/TiO2/O43_0.1z_eps_OUTCAR",
            ],
            "outcar",
        ),
        ("test/data/STO/vasprun.xml", "vasprun.xml"),
    ],
)
def test_load_polarizability_dataset(
    filepaths: str | list[str], file_format: str
) -> None:
    """Test of generic load_polarizability_dataset (normal)."""
    dataset = generic_io.read_polarizability_dataset(filepaths, file_format)
    if isinstance(filepaths, list):
        assert len(dataset) == len(filepaths)
    else:
        assert len(dataset) == 1

    _ = dataset.atomic_numbers
    _ = dataset.mean_polarizability
    _ = dataset.num_atoms
    _ = dataset.num_samples
    _ = dataset.polarizabilities
    _ = dataset.positions


@pytest.mark.parametrize(
    "lattice, atomic_numbers, positions, polarizabilities, exception_type,in_reason",
    [
        (
            np.zeros((3, 3)),
            [1, 2],
            np.random.random((3, 2, 3)),
            np.random.random((2, 3, 3)),
            ValueError,
            "polarizabilities has wrong shape: (2,3,3) != (3,3,3)",
        ),
    ],
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def test_polarizability_dataset_exception(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    polarizabilities: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test polarizability dataset (exception)."""
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        PolarizabilityDataset(lattice, atomic_numbers, positions, polarizabilities)
