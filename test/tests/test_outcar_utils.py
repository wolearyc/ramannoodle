"""Tests for VASP-related routines."""

from typing import TextIO, Type
import numpy as np
from numpy.typing import NDArray

import pytest
import ramannoodle.io.vasp.outcar as vasp_outcar
from ramannoodle.exceptions import InvalidFileException

# pylint: disable=protected-access


@pytest.mark.parametrize(
    "potcar_line, known",
    [
        (" POTCAR:    PAW_PBE Ti_pv 07Sep2000\n", "Ti"),
        (" POTCAR:    PAW_PBE O 08Apr2002  \n", "O"),
    ],
)
def test_get_atomic_symbol_from_potcar_line(potcar_line: str, known: str) -> None:
    """Test get_atomic_symbol_from_potcar_line (normal)."""
    result = vasp_outcar._get_atomic_symbol_from_potcar_line(potcar_line)
    assert result == known


@pytest.mark.parametrize(
    "potcar_line,exception_type,in_reason",
    [
        (
            "bogus_line",
            ValueError,
            "could not parse atomic symbol: bogus_line",
        ),
        (
            "PAW_PBE ZZ_sv 10Sep2004",
            ValueError,
            "unrecognized atomic symbol 'ZZ': PAW_PBE ZZ_sv 10Sep2004",
        ),
        (
            [],
            TypeError,
            "line should have type str, not list",
        ),
    ],
)
def test_get_atomic_symbol_from_potcar_line_exception(
    potcar_line: str,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test get_atomic_symbol_from_potcar_line (exception)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._get_atomic_symbol_from_potcar_line(potcar_line)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_file_fixture, known",
    [
        ("test/data/TiO2/phonons_OUTCAR", ["Ti"] * 36 + ["O"] * 72),
        (
            "test/data/LLZO/LLZO_OUTCAR",
            ["Li"] * 56 + ["La"] * 24 + ["Zr"] * 16 + ["O"] * 96,
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_atomic_symbols_from_outcar(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    known: list[str],
) -> None:
    """Test _read_atomic_symbols_from_outcar (normal)."""
    atomic_symbols = vasp_outcar._read_atomic_symbols(outcar_file_fixture)
    assert atomic_symbols == known


@pytest.mark.parametrize(
    "outcar_file_fixture, exception_type, in_reason",
    [
        (
            "test/data/invalid_vasp/no_elements_OUTCAR",
            InvalidFileException,
            "POTCAR block not found",
        ),
        (
            "test/data/invalid_vasp/no_ion_count_OUTCAR",
            InvalidFileException,
            "ion number block could not be parsed",
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_atomic_symbols_from_outcar_exception(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test _read_atomic_symbols_from_outcar (exception)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._read_atomic_symbols(outcar_file_fixture)
    assert in_reason in str(error.value)


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
    """Test _read_cartesian_positions_from_outcar (normal)."""
    cartesian_positions = vasp_outcar._read_cartesian_positions(
        outcar_file_fixture, 135
    )

    assert len(cartesian_positions) == 135
    assert np.isclose(cartesian_positions[0], known_first_position).all()
    assert np.isclose(cartesian_positions[-1], known_last_position).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, exception_type, in_reason",
    [
        (
            "test/data/invalid_vasp/empty_file",
            InvalidFileException,
            "cartesian positions not found",
        ),
        (
            "test/data/invalid_vasp/invalid_positions_OUTCAR",
            InvalidFileException,
            "cartesian positions could not be parsed",
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_cartesian_positions_from_outcar_exception(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test _read_cartesian_positions_from_outcar (exception)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._read_cartesian_positions(outcar_file_fixture, 20)
    assert in_reason in str(error.value)


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
    """Test _read_fractional_positions_from_outcar (normal)."""
    fractional_positions = vasp_outcar._read_fractional_positions(
        outcar_file_fixture, 135
    )

    assert len(fractional_positions) == 135
    assert np.isclose(fractional_positions[0], known_first_position).all()
    assert np.isclose(fractional_positions[-1], known_last_position).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, exception_type, in_reason",
    [
        (
            "test/data/invalid_vasp/empty_file",
            InvalidFileException,
            "fractional positions not found",
        ),
        (
            "test/data/invalid_vasp/invalid_positions_OUTCAR",
            InvalidFileException,
            "fractional positions could not be parsed",
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_fractional_positions_from_outcar_exception(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test _read_fractional_positions_from_outcar (exception)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._read_fractional_positions(outcar_file_fixture, 20)
    assert in_reason in str(error.value)


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
    """Test _read_polarizability_from_outcar (normal)."""
    polarizability = vasp_outcar._read_polarizability(outcar_file_fixture)

    assert np.isclose(polarizability, known_polarizability).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, exception_type, in_reason",
    [
        (
            "test/data/invalid_vasp/invalid_positions_OUTCAR",
            InvalidFileException,
            "polarizability not found",
        ),
        (
            "test/data/invalid_vasp/invalid_eps_OUTCAR",
            InvalidFileException,
            "polarizability could not be parsed",
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_polarizability_from_outcar_exception(
    outcar_file_fixture: TextIO,  # pylint: disable=redefined-outer-name
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test _read_polarizability_from_outcar (normal)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._read_polarizability(outcar_file_fixture)
    assert in_reason in str(error.value)


@pytest.mark.parametrize(
    "outcar_file_fixture, known_lattice",
    [
        (
            "test/data/TiO2/phonons_OUTCAR",
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
    """Test _read_lattice_from_outcar (normal)."""
    result = vasp_outcar._read_lattice(outcar_file_fixture)
    assert np.isclose(result, known_lattice).all()


@pytest.mark.parametrize(
    "outcar_file_fixture, exception_type, in_reason",
    [
        (
            "test/data/invalid_vasp/empty_file",
            InvalidFileException,
            "outcar does not have expected format",
        ),
        (
            "test/data/invalid_vasp/invalid_positions_lattice_OUTCAR",
            InvalidFileException,
            "lattice could not be parsed:  ",
        ),
    ],
    indirect=["outcar_file_fixture"],
)
def test_read_lattice_from_outcar_exception(
    outcar_file_fixture: TextIO,
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test _read_lattice_from_outcar (exception)."""
    with pytest.raises(exception_type) as error:
        vasp_outcar._read_lattice(outcar_file_fixture)
    assert in_reason in str(error.value)
