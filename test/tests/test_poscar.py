"""Tests for VASP-POSCAR-related routines."""

from typing import Type

import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.io.vasp import poscar
import ramannoodle.io as rn_io


# pylint: disable=protected-access


@pytest.mark.parametrize(
    "atomic_numbers, known_result",
    [([1, 1, 1, 1, 1, 1, 1, 1, 6], "   H  C\n   8   1\n")],
)
def test_get_symbols_str(atomic_numbers: list[int], known_result: str) -> None:
    """Test _get_symbols_str."""
    assert known_result == poscar._get_symbols_str(atomic_numbers)


@pytest.mark.parametrize(
    "atomic_numbers, known_exception, known_reason",
    [
        (
            [1, 1, 1, 1, 1, 1, 1, 1, 6, 2, 1],
            ValueError,
            "atomic number not grouped: 1",
        )
    ],
)
def test_get_symbols_str_exception(
    atomic_numbers: list[int],
    known_exception: Type[Exception],
    known_reason: str,
) -> None:
    """Test _get_symbols_str."""
    with pytest.raises(known_exception) as err:
        poscar._get_symbols_str(atomic_numbers)
    assert known_reason in str(err.value)


@pytest.mark.parametrize(
    "lattice, atomic_numbers, fractional_positions",
    [
        (
            np.array(
                [
                    [4.5359521, 7.92677555, 9.66862649],
                    [6.45429175, 2.36567799, 4.76237761],
                    [3.90418175, 3.25036412, 8.29327069],
                ]
            ),
            [1, 1, 1, 3, 4, 4, 4, 30, 30],
            np.array(
                [
                    [0.53307967, 0.83235714, 0.48246438],
                    [0.76635006, 0.49467758, 0.38580776],
                    [0.78912855, 0.04428546, 0.99963457],
                    [0.48894700, 0.29677130, 0.06598831],
                    [0.95385453, 0.38178553, 0.76467262],
                    [0.34247524, 0.44465406, 0.74414857],
                    [0.81466494, 0.35492304, 0.55763688],
                    [0.46900635, 0.20032024, 0.65554876],
                    [0.61808297, 0.28983299, 0.76844191],
                ]
            ),
        )
    ],
)
def test_write_read_poscar(
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    fractional_positions: NDArray[np.float64],
) -> None:
    """Test write structure as POSCAR (normal)."""
    rn_io.write_structure(
        lattice,
        atomic_numbers,
        fractional_positions,
        "test/data/temp/temp",
        file_format="poscar",
        overwrite="true",
    )


@pytest.mark.parametrize(
    "lattice, atomic_numbers, fractional_positions,path,exception_type,in_reason,"
    "options",
    [
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 30],
            np.zeros((9, 3)),
            "test/data/TiO2",
            FileExistsError,
            "File exists",
            {},
        ),
        (
            np.ones((2, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 30],
            np.zeros((9, 3)),
            "test/data/TiO2",
            ValueError,
            "lattice has wrong shape: (2,3) != (3,3)",
            {},
        ),
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 4],
            np.zeros((9, 3)),
            "test/data/temp",
            ValueError,
            "atomic number not grouped: 4",
            {"overwrite": "true"},
        ),
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30],
            np.zeros((9, 3)),
            "test/data/temp",
            ValueError,
            "positions has wrong shape: (9,3) != (8,3)",
            {"overwrite": "true"},
        ),
    ],
)
def test_write_poscar_exception(  # pylint: disable=too-many-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    fractional_positions: NDArray[np.float64],
    path: str,
    exception_type: Type[Exception],
    in_reason: str,
    options: dict[str, str],
) -> None:
    """Test write structure as POSCAR (exception)."""
    with pytest.raises(exception_type) as err:
        rn_io.write_structure(
            lattice,
            atomic_numbers,
            fractional_positions,
            path,
            file_format="poscar",
            **options,
        )
    assert in_reason in str(err.value)
