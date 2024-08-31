"""Functions for interacting with VASP OUTCAR files."""

from typing import TextIO
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


from ramannoodle.io.io_utils import _skip_file_until_line_contains, pathify
from ramannoodle.exceptions import InvalidFileException, NoMatchingLineFoundException
from ramannoodle.globals import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from ramannoodle.exceptions import get_type_error
from ramannoodle.dynamics.phonon import Phonons
from ramannoodle.dynamics.trajectory import Trajectory
from ramannoodle.structure.reference import ReferenceStructure


# Utilities for OUTCAR. Warning: some of these functions partially read files.
def _get_atomic_symbol_from_potcar_line(line: str) -> str:
    """Read atomic symbol from a POTCAR line in a VASP OUTCAR file.

    Raises
    ------
    TypeError
    ValueError
    """
    try:
        symbol = line.split()[-2].split("_")[0]
    except AttributeError as exc:
        raise get_type_error("line", line, "str") from exc
    except IndexError as exc:
        raise ValueError(f"could not parse atomic symbol: {line}") from exc
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"unrecognized atomic symbol '{symbol}': {line}")
    return symbol


def _read_atomic_symbols(file: TextIO) -> list[str]:
    """Read atomic symbols from a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException

    """
    # Get atom symbols first
    potcar_symbols: list[str] = []
    try:
        line = _skip_file_until_line_contains(file, "POTCAR:    ")
        potcar_symbols.append(_get_atomic_symbol_from_potcar_line(line))
        for line in file:
            if "POTCAR" not in line:
                break
            potcar_symbols.append(_get_atomic_symbol_from_potcar_line(line))

        # HACK: We read the next line and clean up as appropriate.
        # I wish the OUTCAR format was easier to parse, but alas, here we are.
        line = file.readline()
        if "VRHFIN" in line:
            potcar_symbols.pop()  # We read one too many!
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("POTCAR block not found") from exc
    except ValueError as exc:
        raise InvalidFileException(f"POTCAR block could not be parsed: {line}") from exc

    # Then get atom numbers
    try:
        line = _skip_file_until_line_contains(file, "ions per type")
        atomic_numbers = [int(item) for item in line.split()[4:]]

        atomic_symbols = []
        for symbol, number in zip(potcar_symbols, atomic_numbers):
            atomic_symbols += [symbol] * number
        return atomic_symbols
    except (NoMatchingLineFoundException, IndexError) as exc:
        raise InvalidFileException(
            f"ion number block could not be parsed: {line}"
        ) from exc


def _read_eigenvector(
    file: TextIO, num_atoms: int, lattice: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Read the next available phonon eigenvector from a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException
    """
    eigenvector: list[NDArray[np.float64]] = []
    try:
        for line in file:
            if len(eigenvector) == num_atoms:
                break
            if "X" in line:
                continue
            vector = [float(item) for item in line.split()[3:]]
            vector = vector @ np.linalg.inv(lattice)
            eigenvector.append(vector)
        return np.array(eigenvector)
    except (ValueError, IndexError) as exc:
        raise InvalidFileException(f"eigenvector could not be parsed: {line}") from exc


def _read_cart_positions(file: TextIO, num_atoms: int) -> NDArray[np.float64]:
    """Read atomic cartesian positions from a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException
    """
    try:
        _ = _skip_file_until_line_contains(
            file, "position of ions in cartesian coordinates  (Angst):"
        )
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("cartesian positions not found") from exc
    cart_coordinates = []
    for _ in range(num_atoms):
        try:
            line = file.readline()
            cart_coordinates.append([float(item) for item in line.split()[0:3]])
        except (EOFError, ValueError, IndexError) as exc:
            raise InvalidFileException(
                f"cartesian positions could not be parsed: {line}"
            ) from exc

    return np.array(cart_coordinates)


def _read_positions(file: TextIO, num_atoms: int) -> NDArray[np.float64]:
    """Read atomic fractional positions from a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException
    """
    try:
        _ = _skip_file_until_line_contains(
            file, "position of ions in fractional coordinates (direct lattice)"
        )
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("fractional positions not found") from exc

    positions = []
    for _ in range(num_atoms):
        try:
            line = file.readline()
            positions.append([float(item) for item in line.split()[0:3]])
        except (EOFError, ValueError, IndexError) as exc:
            raise InvalidFileException(
                f"fractional positions could not be parsed: {line}"
            ) from exc
    return np.array(positions)


def _read_polarizability(file: TextIO) -> NDArray[np.float64]:
    """Read polarizability from a VASP OUTCAR file.

    In actuality, we read the macroscopic dielectric tensor including local field
    effects.

    Raises
    ------
    InvalidFileException
    """
    try:
        _ = _skip_file_until_line_contains(
            file,
            "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects",
        )
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("polarizability not found") from exc
    file.readline()

    polarizability = []
    for _ in range(3):
        try:
            line = file.readline()
            polarizability.append([float(item) for item in line.split()[0:3]])
        except (EOFError, ValueError, IndexError) as exc:
            raise InvalidFileException(
                f"polarizability could not be parsed: {line}"
            ) from exc
    return np.array(polarizability)


def _get_lattice_vector_from_outcar_line(line: str) -> NDArray[np.float64]:
    """Read lattice vector from an appropriate line in a VASP OUTCAR file.

    Raises
    ------
    TypeError
    ValueError
    """
    try:
        return np.array([float(item) for item in line.split()[0:3]])
    except TypeError as exc:
        raise TypeError("line is not a str") from exc
    except IndexError as exc:
        raise ValueError("line does not have the expected format") from exc


def _read_lattice(file: TextIO) -> NDArray[np.float64]:
    """Read all three lattice vectors (in angstroms) from a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException
    """
    # HACK: if symmetry is turned on, outcar will write additional
    # lattice vectors. We must skip ahead.
    try:
        _ = _skip_file_until_line_contains(
            file,
            "Write flags",
        )
        _ = _skip_file_until_line_contains(
            file,
            "direct lattice vectors      ",
        )
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("outcar does not have expected format") from exc

    lattice = []
    for _ in range(3):
        try:
            line = file.readline()
            lattice.append(_get_lattice_vector_from_outcar_line(line))
        except (EOFError, ValueError) as exc:
            raise InvalidFileException(f"lattice could not be parsed: {line}") from exc

    return np.array(lattice)


def read_phonons(filepath: str | Path) -> Phonons:
    """Read phonons from a VASP OUTCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    :

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    wavenumbers = []
    eigenvectors = []

    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:

        # get atom information
        atomic_symbols = _read_atomic_symbols(file)
        atomic_weights = np.array([ATOMIC_WEIGHTS[symbol] for symbol in atomic_symbols])
        num_atoms = len(atomic_symbols)
        lattice = _read_lattice(file)
        ref_positions = _read_positions(file, len(atomic_symbols))
        num_degrees_of_freedom = num_atoms * 3

        # read in eigenvectors/eigenvalues
        try:
            _ = _skip_file_until_line_contains(
                file, "Eigenvectors and eigenvalues of the dynamical matrix"
            )
        except NoMatchingLineFoundException as exc:
            raise InvalidFileException(
                "eigenvector/eigenvalues block not found"
            ) from exc
        for _ in range(num_degrees_of_freedom):
            try:
                line = _skip_file_until_line_contains(file, "cm-1")
                if "f/i" in line:  # if complex
                    wavenumbers.append(
                        -float(line.split()[6])  # set negative wavenumber
                    )
                else:
                    wavenumbers.append(float(line.split()[7]))
            except (NoMatchingLineFoundException, TypeError, IndexError) as exc:
                raise InvalidFileException("eigenvalue could not be parsed") from exc
            eigenvectors.append(_read_eigenvector(file, num_atoms, lattice))

        # Divide eigenvectors by sqrt(mass) to get cartesian displacements
        wavenumbers = np.array(wavenumbers)
        eigenvectors = np.array(eigenvectors)
        displacements = eigenvectors / np.sqrt(atomic_weights)[:, np.newaxis]

        return Phonons(ref_positions, wavenumbers, displacements)


def read_positions_and_polarizability(
    filepath: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read fractional positions and polarizability from a VASP OUTCAR file.

    The polarizability returned by VASP is, in fact, a dielectric tensor. However,
    this is inconsequential to the calculation of Raman spectra.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        2-tuple:
            0. | positions --
               | (fractional) 2D array with shape (N,3) where N is the number of atoms.
            #. | polarizability --
               | (fractional) 2D array with shape (3,3).

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        positions = read_positions(filepath)
        polarizability = _read_polarizability(file)
        return positions, polarizability


def read_positions(filepath: str | Path) -> NDArray[np.float64]:
    """Read fractional positions from a VASP OUTCAR file.

    Parameters
    ----------
    filepath

    Returns
    -------
    :
        (fractional) 2D array with shape (N,3) where N is the number of atoms.

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        num_atoms = len(_read_atomic_symbols(file))
        positions = _read_positions(file, num_atoms)
        return positions


def read_ref_structure(filepath: str | Path) -> ReferenceStructure:
    """Read reference structure from a VASP OUTCAR file.

    If the file contains multiple structures (such as those generated by a molecular
    dynamics run), the initial structure will be considered the reference structure.

    Parameters
    ----------
    filepath

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    SymmetryException
        Structural symmetry determination failed.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        atomic_symbols = _read_atomic_symbols(file)
        atomic_numbers = [ATOMIC_NUMBERS[symbol] for symbol in atomic_symbols]
        lattice = _read_lattice(file)
        positions = _read_positions(file, len(atomic_symbols))
        return ReferenceStructure(atomic_numbers, lattice, positions)


def _read_next_cart_positions_ts(file: TextIO, num_atoms: int) -> NDArray[np.float64]:
    """Read Cartesian positions contained in a VASP OUTCAR file from an MD run.

    This function assumes that Cartesian positions are ready to be read.

    Raises
    ------
    InvalidFileException
    NoMatchingLineFoundException
    """
    file.readline()
    cart_positions = []
    for _ in range(num_atoms):
        try:
            line = file.readline()
            cart_positions.append([float(item) for item in line.split()[0:3]])
        except (EOFError, ValueError, IndexError) as exc:
            raise InvalidFileException(
                f"Cartesian positions could not be parsed: {line}"
            ) from exc
    return np.array(cart_positions)


def _read_timestep(file: TextIO) -> float:
    """Read timestep (in fs) contained in a VASP OUTCAR file.

    Raises
    ------
    InvalidFileException
    """
    try:
        line = _skip_file_until_line_contains(file, "time-step for ionic-motion")
    except NoMatchingLineFoundException as exc:
        raise InvalidFileException("timestep not found") from exc
    try:
        return float(line.split()[2])
    except (IndexError, TypeError) as exc:
        raise InvalidFileException(f"timestep could not be parsed: {line}") from exc


def read_trajectory(filepath: str | Path) -> Trajectory:
    """Read molecular dynamics trajectory from a VASP OUTCAR file.

    Parameters
    ----------
    filepath

    Raises
    ------
    FileNotFoundError
        File not found.
    InvalidFileException
        Invalid file.
    """
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        atomic_symbols = _read_atomic_symbols(file)
        timestep = _read_timestep(file)
        lattice = _read_lattice(file)

        ml_step = False
        positions_ts = []
        while True:
            try:
                line = _skip_file_until_line_contains(
                    file,
                    "POSITION                                       TOTAL-FORCE ",
                )
            except NoMatchingLineFoundException:
                break

            if ml_step and "(ML)" not in line:
                ml_step = "(ML)" in line
                continue  # we skip ab initio following ML step
            ml_step = "(ML)" in line
            cart_positions = _read_next_cart_positions_ts(file, len(atomic_symbols))
            positions_ts.append(cart_positions @ np.linalg.inv(lattice))

        if len(positions_ts) == 0:
            raise InvalidFileException("no trajectory found")

        return Trajectory(np.array(positions_ts), timestep)
