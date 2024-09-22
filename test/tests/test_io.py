"""Testing for generic io functions."""

from typing import Callable, Type
from pathlib import Path

import pytest
import numpy as np
from numpy.typing import NDArray

import ramannoodle.io.generic as generic_io
from ramannoodle.io.utils import pathify_as_list


@pytest.mark.parametrize(
    "read_function, file_format, reason",
    [
        (generic_io.read_phonons, "bogus_format", "unsupported format: bogus_format"),
        (
            generic_io.read_positions_and_polarizability,
            "bogus_format",
            "unsupported format: bogus_format",
        ),
        (generic_io.read_positions, "bogus_format", "unsupported format: bogus_format"),
        (
            generic_io.read_trajectory,
            "xdatcar",
            "generic.read_trajectory does not support xdatcar",
        ),
        (
            generic_io.read_trajectory,
            "bogus_format",
            "unsupported format: bogus_format",
        ),
        (
            generic_io.read_ref_structure,
            "bogus_format",
            "unsupported format: bogus_format",
        ),
    ],
)
def test_generic_read_exception(
    read_function: Callable[[str | Path, str], None], file_format: str, reason: str
) -> None:
    """Test generic read functions (exception)."""
    with pytest.raises(ValueError) as err:
        read_function("fake path", file_format)
    assert reason in str(err.value)


@pytest.mark.parametrize(
    "lattice, atomic_numbers, positions, file_format, overwrite,exception_type,"
    "in_reason",
    [
        (
            np.zeros((3, 3)),
            [1, 2, 3, 4],
            np.zeros((4, 3)),
            "bogus_format",
            False,
            ValueError,
            "unsupported format: bogus_format",
        ),
        (
            np.zeros((3, 3)),
            [-1, 2, 3, 4],
            np.zeros((4, 3)),
            "poscar",
            False,
            ValueError,
            "invalid atomic number: -1",
        ),
    ],
)
def test_generic_write_structure_exception(  # pylint: disable=too-many-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    file_format: str,
    overwrite: bool,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test generic write_structure (exception)."""
    with pytest.raises(exception_type) as err:
        generic_io.write_structure(
            lattice, atomic_numbers, positions, "fake/path", file_format, overwrite
        )
    assert in_reason in str(err.value)


@pytest.mark.parametrize(
    "filepaths",
    [
        ({"bogus": "filepaths"}),
        (["fake/path", 3.14]),
    ],
)
def test_pathify_exception(filepaths: str | Path | list[str] | list[Path]) -> None:
    """Test pathify (exception)."""
    with pytest.raises(TypeError) as err:
        pathify_as_list(filepaths)
    assert "cannot be resolved as a filepath" in str(err.value)
