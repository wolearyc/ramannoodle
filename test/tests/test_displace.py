"""Tests for structure displacement routines."""

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.vasp as vasp_io
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.structure.displace import write_ast_displaced_structures

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
    """Test write_displaced_structures."""
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
