"""Tests for VASP-POSCAR-related routines."""

from pathlib import Path

import numpy as np

import pytest

import ramannoodle.io.vasp as vasp_io
import ramannoodle.io.generic as generic_io


# pylint: disable=protected-access


@pytest.mark.parametrize(
    "path_fixture, known_num_snapshots",
    [("test/data/STO/XDATCAR", 4)],
    indirect=["path_fixture"],
)
def test_read_write_xdatcar(
    path_fixture: Path,
    known_num_snapshots: int,
) -> None:
    """Test read and writing of XDATCAR (normal)."""
    trajectory = vasp_io.xdatcar.read_trajectory(path_fixture, 1.0)
    assert len(trajectory) == known_num_snapshots

    # First configuration in XDATCAR is identical to poscar.
    ref_structure = vasp_io.poscar.read_ref_structure(path_fixture)
    generic_io.write_trajectory(
        ref_structure.lattice,
        ref_structure.atomic_numbers,
        trajectory.positions_ts,
        "test/data/temp",
        file_format="xdatcar",
        overwrite=True,
    )
    written_trajectory = vasp_io.xdatcar.read_trajectory("test/data/temp", 1.0)
    assert np.isclose(written_trajectory.positions_ts, trajectory.positions_ts).all()
    assert written_trajectory.timestep == trajectory.timestep
