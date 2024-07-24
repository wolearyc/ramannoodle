"""Testing for the VASP utilities"""

from pathlib import Path
from typing import TextIO
import numpy as np
from numpy.typing import NDArray

import pytest

from ramannoodle.io.vasp import vasp_utils, load_phonons_from_outcar
from .. import EPS_OUTCAR_NUM_ATOMS


def test_read_outcar(
    outcar_path_fixture: Path,
) -> None:
    """Tests outcar reading"""
    phonons = load_phonons_from_outcar(outcar_path_fixture)
    assert len(phonons.get_wavenumbers()) == len(phonons.get_displacements()) == 321


@pytest.mark.parametrize(
    "potcar_line, expected",
    [
        (" POTCAR:    PAW_PBE Ti_pv 07Sep2000\n", "Ti"),
        (" POTCAR:    PAW_PBE O 08Apr2002  \n", "O"),
    ],
)
def test_get_atom_symbol_from_potcar_line(potcar_line: str, expected: str) -> None:
    """test"""
    result = vasp_utils._get_atom_symbol_from_potcar_line(  # pylint: disable=protected-access
        potcar_line
    )
    assert result == expected


def test_read_atom_symbols_from_outcar(
    phonon_outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
) -> None:
    """test"""
    atom_symbols = (
        vasp_utils._read_atom_symbols_from_outcar(  # pylint: disable=protected-access
            phonon_outcar_file_fixture
        )
    )
    assert atom_symbols == ["Ti"] * 36 + ["O"] * 72


@pytest.mark.parametrize(
    "eps_outcar_file_fixture, known_first_position, known_last_position",
    [
        (
            "eps_outcar_file_fixture",
            np.array([11.82301433, 0.00141878, 11.82095340]),
            np.array([7.88377093, 9.85727498, 9.86042313]),
        ),
    ],
    indirect=["eps_outcar_file_fixture"],
)
def test_read_cartesian_positions_from_outcar(
    eps_outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known_first_position: NDArray[np.float64],
    known_last_position: NDArray[np.float64],
) -> None:
    """test"""
    cartesian_positions = vasp_utils._read_cartesian_positions_from_outcar(  # pylint: disable=protected-access
        eps_outcar_file_fixture, EPS_OUTCAR_NUM_ATOMS
    )

    assert len(cartesian_positions) == EPS_OUTCAR_NUM_ATOMS
    assert np.isclose(cartesian_positions[0], known_first_position).all()
    assert np.isclose(cartesian_positions[-1], known_last_position).all()


@pytest.mark.parametrize(
    "eps_outcar_file_fixture, known_polarizability, ",
    [
        (
            "eps_outcar_file_fixture",
            np.array(
                [
                    [5.704647, -0.000011, 0.000010],
                    [-0.000007, 5.704472, -0.000017],
                    [0.000008, -0.000022, 5.704627],
                ]
            ),
        ),
    ],
    indirect=["eps_outcar_file_fixture"],
)
def test_read_polarizability_from_outcar(
    eps_outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known_polarizability: NDArray[np.float64],
) -> None:
    """test"""

    polarizability = (
        vasp_utils._read_polarizability_from_outcar(  # pylint: disable=protected-access
            eps_outcar_file_fixture
        )
    )

    assert np.isclose(polarizability, known_polarizability).all()
