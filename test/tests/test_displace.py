"""Tests for structure displacement functions."""

from typing import Type
import re

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.vasp as vasp_io
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.structure.displace import (
    write_ast_displaced_structures,
    get_ast_displaced_positions,
    write_displaced_structures,
)

# pylint: disable=protected-access


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_index,cart_direction,amplitude,outcar_known",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            42,
            np.array([0.0, 0, 1]),
            0.1,
            "test/data/TiO2/O43_0.1z_eps_OUTCAR",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            4,
            np.array([10, 0, 0]),
            -0.2,
            "test/data/TiO2/Ti5_m0.2x_eps_OUTCAR",
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_write_ast_displaced_structures(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitude: float,
    outcar_known: str,
) -> None:
    """Test write_ast_displaced_structures."""
    amplitudes = np.array([amplitude])
    write_ast_displaced_structures(
        outcar_ref_structure_fixture,
        atom_index,
        cart_direction,
        amplitudes,
        ["test/data/temp.vasp"],
        "poscar",
        overwrite=True,
    )

    known_positions = vasp_io.outcar.read_positions(outcar_known)
    assert np.isclose(
        vasp_io.poscar.read_positions("test/data/temp.vasp"), known_positions
    ).all()


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,cart_displacement, amplitude, poscar_known",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            np.ones((108, 3)),
            0.05 * 11.3768434223423398,
            "test/data/TiO2/displaced_POSCAR",
        )
    ],
    indirect=["outcar_ref_structure_fixture"],
)
def test_write_displaced_structures(
    outcar_ref_structure_fixture: ReferenceStructure,
    cart_displacement: NDArray[np.float64],
    amplitude: float,
    poscar_known: str,
) -> None:
    """Test write_displaced_structures."""
    amplitudes = np.array([amplitude])
    write_displaced_structures(
        outcar_ref_structure_fixture,
        cart_displacement,
        amplitudes,
        ["test/data/temp.vasp"],
        "poscar",
        overwrite=True,
    )

    known_positions = vasp_io.poscar.read_positions(poscar_known)
    assert np.isclose(
        vasp_io.poscar.read_positions("test/data/temp.vasp"), known_positions
    ).all()


@pytest.mark.parametrize(
    "outcar_ref_structure_fixture,atom_index,cart_direction,amplitude,exception_type,"
    "in_reason",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
            42,
            {},
            0.1,
            TypeError,
            "cart_direction should have type ndarray, not dict",
        ),
        (
            "test/data/TiO2/phonons_OUTCAR",
            300,
            np.array([0, 0, 1]),
            0.1,
            IndexError,
            "invalid atom_index: 300",
        ),
    ],
    indirect=["outcar_ref_structure_fixture"],
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def test_get_ast_displaced_positions_exception(
    outcar_ref_structure_fixture: ReferenceStructure,
    atom_index: int,
    cart_direction: NDArray[np.float64],
    amplitude: float,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test get_ast_displaced_positions (exception)."""
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        amplitudes = np.array([amplitude])
        get_ast_displaced_positions(
            outcar_ref_structure_fixture,
            atom_index,
            cart_direction,
            amplitudes,
        )
