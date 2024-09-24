"""Tests for VASP vasprun.xml functions."""

from typing import Type, Callable
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

import pytest

import ramannoodle.io.generic as generic_io
import ramannoodle.io.vasp as vasp_io
from ramannoodle.exceptions import InvalidFileException


# pylint: disable=protected-access


@pytest.mark.parametrize(
    "path_fixture, known_last_positions, known_polarizability",
    [
        (
            "test/data/STO/vasprun.xml",
            np.array([0.83333331, 0.83333331, 0.29210141]),
            np.array(
                [
                    [6.95882353e00, -5.75000000e-06, -8.96000000e-06],
                    [-2.90000000e-07, 6.95884757e00, -5.35000000e-06],
                    [-4.13000000e-06, -6.04000000e-06, 6.37287749e00],
                ]
            ),
        )
    ],
    indirect=["path_fixture"],
)
def test_read_positions_and_polarizability(
    path_fixture: Path,
    known_last_positions: NDArray[np.float64],
    known_polarizability: NDArray[np.float64],
) -> None:
    """Test read_positions_and_polarizability (normal)."""
    positions, polarizability = generic_io.read_positions_and_polarizability(
        path_fixture, file_format="vasprun.xml"
    )

    assert np.allclose(positions[-1], known_last_positions)
    assert np.allclose(polarizability, known_polarizability)


@pytest.mark.parametrize(
    "path_fixture, known_last_positions",
    [
        (
            "test/data/STO/vasprun.xml",
            np.array([0.83333331, 0.83333331, 0.29210141]),
        )
    ],
    indirect=["path_fixture"],
)
def test_read_positions(
    path_fixture: Path,
    known_last_positions: NDArray[np.float64],
) -> None:
    """Test read_positions (normal)."""
    positions = generic_io.read_positions(path_fixture, file_format="vasprun.xml")

    assert np.allclose(positions[-1], known_last_positions)


@pytest.mark.parametrize(
    "path_fixture, known_atomic_numbers, known_lattice",
    [
        (
            "test/data/STO/vasprun.xml",
            [22] * 36 + [8] * 72,
            np.array(
                [
                    [11.37684345, 0.0, 0.0],
                    [0.0, 11.37684345, 0.0],
                    [0.0, 0.0, 9.6045742],
                ]
            ),
        )
    ],
    indirect=["path_fixture"],
)
def test_read_ref_structure(
    path_fixture: Path,
    known_atomic_numbers: list[int],
    known_lattice: NDArray[np.float64],
) -> None:
    """Test read_ref_structure (normal)."""
    ref_structure = generic_io.read_ref_structure(
        path_fixture, file_format="vasprun.xml"
    )

    assert ref_structure.atomic_numbers == known_atomic_numbers
    assert np.allclose(ref_structure.lattice, known_lattice)


@pytest.mark.parametrize(
    "path_fixture, known_final_position, known_trajectory_length, known_timestep",
    [
        (
            "test/data/TiO2/md_run_vasprun.xml",
            np.array([0.83414850, 0.82850374, 0.30051845]),
            19,
            1,
        )
    ],
    indirect=["path_fixture"],
)
def test_read_trajectory(
    path_fixture: Path,
    known_final_position: NDArray[np.float64],
    known_trajectory_length: int,
    known_timestep: float,
) -> None:
    """Test read_ref_structure (normal)."""
    trajectory = generic_io.read_trajectory(path_fixture, file_format="vasprun.xml")

    assert np.allclose(trajectory.positions_ts[-1][-1], known_final_position)
    assert len(trajectory) == known_trajectory_length
    assert np.isclose(trajectory.timestep, known_timestep)


@pytest.mark.parametrize(
    "path_fixture, known_wavenumbers, known_last_displacement",
    [
        (
            "test/data/STO/phonons_vasprun.xml",
            np.array([827.784966, 796.172156, 796.172156, 796.172156, 749.152843]),
            np.array([-0.000000, 0.011695, 0.000227] / np.sqrt(15.999) / 7.88204464),
        )
    ],
    indirect=["path_fixture"],
)
def test_read_phonons(
    path_fixture: Path,
    known_wavenumbers: NDArray[np.float64],
    known_last_displacement: NDArray[np.float64],
) -> None:
    """Test read_phonons (normal)."""
    phonons = generic_io.read_phonons(path_fixture, file_format="vasprun.xml")
    degrees_of_freedom = len(phonons.ref_positions) * 3

    assert len(phonons.wavenumbers) == len(phonons.displacements)
    assert len(phonons.wavenumbers) == degrees_of_freedom
    assert np.allclose(phonons.wavenumbers[0:5], known_wavenumbers)
    assert np.allclose(phonons.displacements[-1][-1], known_last_displacement)


@pytest.mark.parametrize(
    "read_function, path_fixture, exception_type, in_reason",
    [
        (
            vasp_io.vasprun.read_positions,
            "test/data/TiO2/POSCAR",
            InvalidFileException,
            "root xml element could not be found",
        ),
        (
            vasp_io.vasprun.read_positions_and_polarizability,
            "test/data/malformed/vasprun.xml",
            InvalidFileException,
            "initial positions not found",
        ),
        (
            vasp_io.vasprun.read_positions,
            "test/data/malformed/vasprun.xml",
            InvalidFileException,
            "initial positions not found",
        ),
        (
            vasp_io.vasprun.read_phonons,
            "test/data/malformed/vasprun.xml",
            InvalidFileException,
            "atomic symbols not found",
        ),
        (
            vasp_io.vasprun.read_ref_structure,
            "test/data/malformed/vasprun.xml",
            InvalidFileException,
            "atomic symbols not found",
        ),
        (
            vasp_io.vasprun.read_trajectory,
            "test/data/malformed/vasprun.xml",
            InvalidFileException,
            "no trajectory found",
        ),
    ],
    indirect=["path_fixture"],
)
def test_read_vasprun_exception(
    read_function: Callable[[str | Path], None],
    path_fixture: Path,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test vasprun read functions (exception)."""
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        read_function(path_fixture)
