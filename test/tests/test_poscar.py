"""Tests for VASP-POSCAR-related routines."""

from typing import Type
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.generic
import ramannoodle.io.vasp as vasp_io
from ramannoodle.io.vasp import poscar
from ramannoodle.exceptions import InvalidFileException


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
    "lattice, atomic_numbers, positions",
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
    positions: NDArray[np.float64],
) -> None:
    """Test write structure as POSCAR (normal)."""
    ramannoodle.io.generic.write_structure(
        lattice,
        atomic_numbers,
        positions,
        "test/data/temp",
        file_format="poscar",
        overwrite=True,
    )
    reference_structure = ramannoodle.io.generic.read_ref_structure(
        "test/data/temp", file_format="poscar"
    )
    assert np.isclose(reference_structure._lattice, lattice).all()
    assert np.isclose(reference_structure._atomic_numbers, atomic_numbers).all()
    assert np.isclose(reference_structure.positions, positions).all()


@pytest.mark.parametrize(
    "lattice, atomic_numbers, positions,path,exception_type,in_reason,overwrite",
    [
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 30],
            np.zeros((9, 3)),
            "test/data/TiO2",
            FileExistsError,
            "File exists",
            False,
        ),
        (
            np.ones((2, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 30],
            np.zeros((9, 3)),
            "test/data/temp",
            ValueError,
            "lattice has wrong shape: (2,3) != (3,3)",
            False,
        ),
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30, 4],
            np.zeros((9, 3)),
            "test/data/temp",
            ValueError,
            "atomic number not grouped: 4",
            True,
        ),
        (
            np.ones((3, 3)),
            [1, 1, 1, 3, 4, 4, 4, 30],
            np.zeros((9, 3)),
            "test/data/temp",
            ValueError,
            "positions has wrong shape: (9,3) != (8,3)",
            True,
        ),
    ],
)
def test_write_poscar_exception(  # pylint: disable=too-many-arguments
    lattice: NDArray[np.float64],
    atomic_numbers: list[int],
    positions: NDArray[np.float64],
    path: str,
    exception_type: Type[Exception],
    in_reason: str,
    overwrite: bool,
) -> None:
    """Test write structure as POSCAR (exception)."""
    with pytest.raises(exception_type) as err:
        ramannoodle.io.generic.write_structure(
            lattice, atomic_numbers, positions, path, "poscar", overwrite
        )
    assert in_reason in str(err.value)


@pytest.mark.parametrize(
    "cart_poscar_path, ref_frac_poscar_path",
    [("test/data/TiO2/cart_POSCAR", "test/data/TiO2/POSCAR")],
)
def test_read_cart_poscar(  # pylint: disable=too-many-arguments
    cart_poscar_path: str, ref_frac_poscar_path: str
) -> None:
    """Test write structure as POSCAR (exception)."""
    ref_structure = vasp_io.poscar.read_ref_structure(cart_poscar_path)
    known_ref_structure = vasp_io.poscar.read_ref_structure(ref_frac_poscar_path)

    assert np.isclose(ref_structure.lattice, known_ref_structure.lattice).all()
    assert np.isclose(ref_structure.positions, known_ref_structure.positions).all()


@pytest.mark.parametrize(
    "path_fixture, exception_type, in_reason",
    [
        (
            "test/data/malformed/vasp.poscar/bogus_label",
            InvalidFileException,
            "unrecognized coordinate format: hello world!",
        ),
        (
            "test/data/malformed/vasp.poscar/missing_atoms",
            InvalidFileException,
            "positions could not be parsed:",
        ),
        (
            "test/data/malformed/vasp.poscar/missing_basis",
            InvalidFileException,
            "lattice could not be parsed:",
        ),
        (
            "test/data/malformed/vasp.poscar/missing_count",
            InvalidFileException,
            "wrong number of ion counts: 1 != 2",
        ),
        (
            "test/data/malformed/vasp.poscar/missing_ion",
            InvalidFileException,
            "wrong number of ion counts: 2 != 1",
        ),
        (
            "test/data/malformed/vasp.poscar/missing_label",
            InvalidFileException,
            "scale factor could not be parsed:",
        ),
        (
            "test/data/malformed/vasp.poscar/too_many_labels",
            InvalidFileException,
            "positions could not be parsed: This shouldn't be here",
        ),
        (
            "test/data/malformed/vasp.poscar/bogus_element",
            InvalidFileException,
            "unrecognized atom symbol: Imaginarium",
        ),
        (
            "test/data/malformed/vasp.poscar/no_elements",
            InvalidFileException,
            "no atom symbols found",
        ),
        (
            "test/data/malformed/vasp.poscar/bogus_ion_count",
            InvalidFileException,
            "could not parse counts:",
        ),
    ],
    indirect=["path_fixture"],
)
def test_read_poscar_exception(
    path_fixture: Path,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test poscar reading (exception)."""
    with pytest.raises(exception_type) as error:
        ramannoodle.io.generic.read_ref_structure(path_fixture, file_format="poscar")
    assert in_reason in str(error.value)
