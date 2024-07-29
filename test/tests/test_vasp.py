"""Testing for the VASP utilities"""

from pathlib import Path
from typing import TextIO
import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.io.vasp import (
    vasp_utils,
    load_phonons_from_outcar,
)
from ramannoodle.globals import ATOMIC_WEIGHTS

from .. import EPS_OUTCAR_NUM_ATOMS, PHONONS_OUTCAR_NUM_ATOMS

# pylint: disable=protected-access


@pytest.mark.parametrize(
    "outcar_path_fixture, known_num_atoms, known_wavenumbers,"
    "known_first_displacement, known_last_displacement",
    [
        (
            "test/data/TiO2_OUTCAR",
            PHONONS_OUTCAR_NUM_ATOMS,
            np.array([811.691808, 811.691808, 811.691808, 811.691808]),
            np.array([-0.068172, 0.046409, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["Ti"]),
            np.array([-0.011752, 0.074105, 0.000000]) / np.sqrt(ATOMIC_WEIGHTS["O"]),
        ),
    ],
    indirect=["outcar_path_fixture"],
)
def test_load_phonons_from_outcar(
    outcar_path_fixture: Path,
    known_num_atoms: int,
    known_wavenumbers: NDArray[np.float64],
    known_first_displacement: NDArray[np.float64],
    known_last_displacement: NDArray[np.float64],
) -> None:
    """Tests outcar reading"""
    phonons = load_phonons_from_outcar(outcar_path_fixture)

    known_degrees_of_freedom = known_num_atoms * 3
    assert phonons.get_wavenumbers().shape == (known_degrees_of_freedom,)
    assert np.isclose(phonons.get_wavenumbers()[0:4], known_wavenumbers).all()
    assert phonons.get_displacements().shape == (
        known_degrees_of_freedom,
        known_num_atoms,
        3,
    )
    assert np.isclose(phonons.get_displacements()[0, 0], known_first_displacement).all()
    print(phonons.get_displacements()[-1, -1])
    assert np.isclose(
        phonons.get_displacements()[-1, -1], known_last_displacement
    ).all()


@pytest.mark.parametrize(
    "potcar_line, known",
    [
        (" POTCAR:    PAW_PBE Ti_pv 07Sep2000\n", "Ti"),
        (" POTCAR:    PAW_PBE O 08Apr2002  \n", "O"),
    ],
)
def test_get_atomic_symbol_from_potcar_line(potcar_line: str, known: str) -> None:
    """test"""
    result = vasp_utils._get_atomic_symbol_from_potcar_line(potcar_line)
    assert result == known


@pytest.mark.parametrize(
    "outcar_file_fixture, known",
    [("test/data/TiO2_OUTCAR", ["Ti"] * 36 + ["O"] * 72)],
    indirect=["outcar_file_fixture"],
)
def test_read_atomic_symbols_from_outcar(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known: list[str],
) -> None:
    """test"""
    atomic_symbols = vasp_utils._read_atomic_symbols_from_outcar(outcar_file_fixture)
    assert atomic_symbols == known


@pytest.mark.parametrize(
    "outcar_file_fixture, known_first_position, known_last_position",
    [
        (
            "test/data/EPS_OUTCAR",
            np.array([11.82301433, 0.00141878, 11.82095340]),
            np.array([7.88377093, 9.85727498, 9.86042313]),
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_cartesian_positions_from_outcar(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known_first_position: NDArray[np.float64],
    known_last_position: NDArray[np.float64],
) -> None:
    """test"""
    cartesian_positions = vasp_utils._read_cartesian_positions_from_outcar(
        outcar_file_fixture, EPS_OUTCAR_NUM_ATOMS
    )

    assert len(cartesian_positions) == EPS_OUTCAR_NUM_ATOMS
    assert np.isclose(cartesian_positions[0], known_first_position).all()
    assert np.isclose(cartesian_positions[-1], known_last_position).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, known_first_position, known_last_position",
    [
        (
            "test/data/EPS_OUTCAR",
            np.array([0.999995547, 0.000120001, 0.999821233]),
            np.array([0.666812676, 0.833732482, 0.833998755]),
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_fractional_positions_from_outcar(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known_first_position: NDArray[np.float64],
    known_last_position: NDArray[np.float64],
) -> None:
    """test"""
    fractional_positions = vasp_utils._read_fractional_positions_from_outcar(
        outcar_file_fixture, EPS_OUTCAR_NUM_ATOMS
    )

    assert len(fractional_positions) == EPS_OUTCAR_NUM_ATOMS
    assert np.isclose(fractional_positions[0], known_first_position).all()
    assert np.isclose(fractional_positions[-1], known_last_position).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, known_polarizability, ",
    [
        (
            "test/data/EPS_OUTCAR",
            np.array(
                [
                    [5.704647, -0.000011, 0.000010],
                    [-0.000007, 5.704472, -0.000017],
                    [0.000008, -0.000022, 5.704627],
                ]
            ),
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_polarizability_from_outcar(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known_polarizability: NDArray[np.float64],
) -> None:
    """test"""

    polarizability = vasp_utils._read_polarizability_from_outcar(outcar_file_fixture)

    assert np.isclose(polarizability, known_polarizability).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, known_lattice",
    [
        (
            "test/data/TiO2_OUTCAR",
            np.array(
                [
                    [11.3768434223, 0.0000000000, 0.0000000000],
                    [0.0000000000, 11.3768434223, 0.0000000000],
                    [0.0000000000, 0.0000000000, 9.6045745743],
                ]
            ),
        ),
        (
            "test/data/EPS_OUTCAR",
            np.array(
                [
                    [11.823066970, 0.000000000, 0.000000000],
                    [0.000000000, 11.823066970, 0.000000000],
                    [0.000000000, 0.000000000, 11.823066970],
                ]
            ),
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_lattice_from_outcar(
    outcar_file_fixture: TextIO, known_lattice: NDArray[np.float64]
) -> None:
    """test"""
    result = vasp_utils._read_lattice_from_outcar(outcar_file_fixture)
    assert np.isclose(result, known_lattice).all()
