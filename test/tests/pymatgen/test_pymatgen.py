"""Tests for pymatgen IO functions."""

from pathlib import Path

import pytest


import numpy as np

import pymatgen.core
import pymatgen.core.trajectory
import pymatgen.io.vasp

import ramannoodle as rn
import ramannoodle.io.pymatgen as pymatgen_io

# pylint: disable=protected-access


@pytest.mark.parametrize("path_fixture", ["test/data/TiO2/POSCAR"])
def test_get_positions(path_fixture: Path) -> None:
    """Test get_positions (normal)."""
    known_positions = rn.io.vasp.poscar.read_positions(path_fixture)

    positions = pymatgen_io.get_positions(
        pymatgen.core.Structure.from_file(path_fixture)
    )
    assert np.allclose(known_positions, positions)


@pytest.mark.parametrize("path_fixture", ["test/data/TiO2/POSCAR"])
def test_construct_ref_structure(path_fixture: Path) -> None:
    """Test construct_ref_structure (normal)."""
    known_ref_structure = rn.io.vasp.poscar.read_ref_structure(path_fixture)

    structure = pymatgen.core.Structure.from_file(path_fixture)
    ref_structure = pymatgen_io.construct_ref_structure(structure)

    assert ref_structure.atomic_numbers == known_ref_structure.atomic_numbers


@pytest.mark.parametrize("path_fixture", ["test/data/TiO2/POSCAR"])
def test_get_structure(path_fixture: Path) -> None:
    """Test get_structure (normal)."""
    known_structure = rn.io.vasp.poscar.read_structure(path_fixture)

    pymatgen_structure = pymatgen.core.Structure.from_file(path_fixture)
    structure = pymatgen_io.get_structure(pymatgen_structure)

    assert np.allclose(known_structure[0], structure[0])
    assert known_structure[1] == structure[1]
    assert np.allclose(known_structure[2], structure[2])


@pytest.mark.parametrize("path_fixture", ["test/data/STO/XDATCAR"])
def test_construct_trajectory(path_fixture: Path) -> None:
    """Test construct_trajectory (normal)."""
    known_trajectory = rn.io.vasp.xdatcar.read_trajectory(path_fixture, 1.0)

    pymatgen_trajectory = pymatgen.core.trajectory.Trajectory.from_file(path_fixture)
    trajectory = pymatgen_io.construct_trajectory(pymatgen_trajectory, 1.0)

    assert np.allclose(known_trajectory.positions_ts, trajectory.positions_ts)


@pytest.mark.parametrize(
    "filepaths",
    [
        [
            "test/data/STO/vasprun.xml",
        ],
    ],
)
def test_load_polarizability_dataset(filepaths: str | list[str]) -> None:
    """Test of construct_polarizability_dataset (normal)."""
    known_dataset = rn.io.generic.read_polarizability_dataset(filepaths, "vasprun.xml")

    structures = []
    polarizabilities = []
    for filepath in filepaths:
        vasprun = pymatgen.io.vasp.outputs.Vasprun(filepath)
        polarizabilities.append(np.array(np.array(vasprun.epsilon_static)))
        pymatgen_structure = pymatgen.core.Structure.from_file(filepath)

        structures.append(pymatgen_structure)

    dataset = pymatgen_io.construct_polarizability_dataset(
        structures, np.array(polarizabilities)
    )

    assert np.allclose(known_dataset.polarizabilities, dataset.polarizabilities)
