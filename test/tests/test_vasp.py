"""Testing for the VASP utilities"""

from pathlib import Path
from typing import TextIO

import pytest

from ramannoodle.dynamics import vasp_utils


def test_read_outcar(
    outcar_path_fixture: Path,
) -> None:
    """Tests outcar reading"""
    phonons = vasp_utils.load_phonons_from_outcar(outcar_path_fixture)
    assert len(phonons.get_wavenumbers()) == len(phonons.get_displacements()) == 321


@pytest.mark.parametrize(
    'potcar_line, expected',
    [
        (' POTCAR:    PAW_PBE Ti_pv 07Sep2000\n', 'Ti'),
        (' POTCAR:    PAW_PBE O 08Apr2002  \n', 'O'),
    ],
)
def test_get_atom_symbol_from_potcar_line(potcar_line: str, expected: str) -> None:
    """test"""
    result = vasp_utils._get_atom_symbol_from_potcar_line(  # pylint: disable=protected-access
        potcar_line
    )
    assert result == expected


def test_read_atom_symbols_from_outcar(
    outcar_file_fixture: TextIO,
) -> None:
    """test"""
    atom_symbols = (
        vasp_utils._read_atom_symbols_from_outcar(  # pylint: disable=protected-access
            outcar_file_fixture
        )
    )
    assert atom_symbols == ['Ti'] * 36 + ['O'] * 72
