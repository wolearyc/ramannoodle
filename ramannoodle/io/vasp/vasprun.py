"""Functions for interacting with vasprun.xml files."""

from typing import TextIO
from pathlib import Path
from xml.etree.ElementTree import Element
import defusedxml.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from ramannoodle.io.io_utils import pathify
from ramannoodle.exceptions import InvalidFileException
from ramannoodle.globals import ATOMIC_WEIGHTS, ATOMIC_NUMBERS
from ramannoodle.dynamics.phonon import Phonons
from ramannoodle.dynamics.trajectory import Trajectory
from ramannoodle.structure.reference import ReferenceStructure


def _get_root_element(file: TextIO) -> Element:
    try:
        root = ET.parse(file).getroot()
        assert isinstance(root, Element)
        return root
    except ET.ParseError as exc:
        raise InvalidFileException("root xml element could not be found") from exc


def _parse_atomic_symbols(root: Element) -> list[str]:
    """Parse atomic symbols from a vasprun.xml Element.

    Raises
    ------
    InvalidFileException

    """
    set_element = root.find("./atominfo/array/set")
    if set_element is None:
        raise InvalidFileException("atomic symbols not found")

    atomic_symbols = []
    for child in set_element:
        atomic_symbol = child[0].text
        if atomic_symbol is None:
            raise InvalidFileException("child has no text")
        atomic_symbols.append(atomic_symbol.strip())
    return atomic_symbols


def _parse_positions(structure_varray: Element) -> NDArray[np.float64]:
    """Parse atomic fractional positions from a vasprun.xml Element.

    This function reads the positions from the initial structure.

    Raises
    ------
    InvalidFileException
    """
    positions = []
    for child in structure_varray:
        text = child.text
        if text is None:
            raise InvalidFileException("varray child text not found")
        positions.append([float(i) for i in text.split()])
    return np.array(positions)


def _parse_polarizability(root: Element) -> NDArray[np.float64]:
    """Parse polarizability from a vasprun.xml Element.

    In actuality, we read the macroscopic dielectric tensor including local field
    effects.

    Raises
    ------
    InvalidFileException
    """
    polarizability_element = root.find("./calculation/varray[@name='dielectric_dft']")
    if polarizability_element is None:
        raise InvalidFileException("polarizability not found")
    polarizability = []
    for child in polarizability_element:
        text = child.text
        if text is None:
            raise InvalidFileException("varray child text not found")
        polarizability.append([float(i) for i in text.split()])
    return np.array(polarizability)


def _parse_lattice(root: Element) -> NDArray[np.float64]:
    """Parse all three lattice vectors (in angstroms) from a vasprun.xml Element.

    Raises
    ------
    InvalidFileException
    """
    varray_element = root.find(
        "./structure[@name='initialpos']/crystal/varray[@name='basis']"
    )
    if varray_element is None:
        raise InvalidFileException("lattice not found")

    lattice = []
    for child in varray_element:
        text = child.text
        if text is None:
            raise InvalidFileException("varray child text not found")
        lattice.append([float(i) for i in text.split()])
    return np.array(lattice)


def read_positions_and_polarizability(
    filepath: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read fractional positions and polarizability from a vasprun.xml file.

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
        root = _get_root_element(file)
        structure_varray = root.find("./structure[@name='initialpos']/varray")
        if structure_varray is None:
            raise InvalidFileException("initial positions not found")
        positions = _parse_positions(structure_varray)
        polarizability = _parse_polarizability(root)
        return positions, polarizability


def read_positions(filepath: str | Path) -> NDArray[np.float64]:
    """Read fractional positions from a vasprun.xml file.

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
        root = _get_root_element(file)
        structure_varray = root.find("./structure[@name='initialpos']/varray")
        if structure_varray is None:
            raise InvalidFileException("initial positions not found")
        positions = _parse_positions(structure_varray)
        return positions


def read_ref_structure(filepath: str | Path) -> ReferenceStructure:
    """Read reference structure from a vasprun.xml file.

    If the file contains multiple structures (such as those generated by a
    molecular dynamics run), the initial structure will be considered the reference
    structure.

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
        root = _get_root_element(file)
        atomic_symbols = _parse_atomic_symbols(root)
        atomic_numbers = [ATOMIC_NUMBERS[symbol] for symbol in atomic_symbols]
        lattice = _parse_lattice(root)
        structure_varray = root.find("./structure[@name='initialpos']/varray")
        if structure_varray is None:
            raise InvalidFileException("initial positions not found")
        positions = _parse_positions(structure_varray)
        return ReferenceStructure(atomic_numbers, lattice, positions)


def _parse_timestep(root: Element) -> float:
    """Parse timestep in fs from a vasprun.xml Element.

    Raises
    ------
    InvalidFileException
    """
    element = root.find("./parameters/separator[@name='ionic']/i/[@name='POTIM']")
    if element is None:
        raise InvalidFileException("timestep not found")
    text = element.text
    if text is None:
        raise InvalidFileException("potim element has no text")
    return float(text.strip())


def read_trajectory(filepath: str | Path) -> Trajectory:
    """Read molecular dynamics trajectory from a vasprun.xml file.

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
        root = _get_root_element(file)

        structures = root.iterfind("structure")
        positions_ts = []
        for structure in structures:
            if "name" in structure.attrib:  # skip named structures
                continue
            structure_varray = structure.find("varray")
            if structure_varray is None:
                raise InvalidFileException("structure varray not found")
            positions_ts.append(_parse_positions(structure_varray))
        if len(positions_ts) == 0:
            raise InvalidFileException("no trajectory found")

        timestep = _parse_timestep(root)

        return Trajectory(np.array(positions_ts), timestep)


def _parse_eigenvalues(root: Element) -> NDArray[np.float64]:
    """Parse eigenvalues in cm-1 from a vasprun.xml Element.

    Raises
    ------
    InvalidFileException
    """
    element = root.find("./calculation/dynmat/v[@name='eigenvalues']")
    if element is None:
        raise InvalidFileException("eigenvalues not found")
    if element.text is None:
        raise InvalidFileException("eigenvalues has no text")
    eigenvalues = np.array([float(i) for i in element.text.split()])
    eigenvalues = np.sqrt(np.abs(-eigenvalues)) * 1 / 0.0299792458
    eigenvalues *= np.sign(eigenvalues)
    return eigenvalues


def _parse_eigenvectors(root: Element) -> NDArray[np.float64]:
    """Parse eigenvectors in cm-1 from a vasprun.xml Element.

    Raises
    ------
    InvalidFileException
    """
    lattice = _parse_lattice(root)
    num_atoms = len(_parse_atomic_symbols(root))
    element = root.find("./calculation/dynmat/varray[@name='eigenvectors']")
    if element is None:
        raise InvalidFileException("eigenvectors not found")

    eigenvectors = []
    for v in element:
        if v.text is None:
            raise InvalidFileException("eigenvector element has no text")
        cart_eigenvector = np.array([float(i) for i in v.text.split()])
        cart_eigenvector = cart_eigenvector.reshape((num_atoms, 3))
        eigenvectors.append(cart_eigenvector @ np.linalg.inv(lattice))

    return np.array(eigenvectors)


def read_phonons(filepath: str | Path) -> Phonons:
    """Read phonons from a vasprun.xml file.

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
    filepath = pathify(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        root = _get_root_element(file)

        atomic_symbols = _parse_atomic_symbols(root)
        atomic_weights = np.array([ATOMIC_WEIGHTS[symbol] for symbol in atomic_symbols])
        structure_varray = root.find("./structure[@name='initialpos']/varray")
        if structure_varray is None:
            raise InvalidFileException("initial positions not found")
        ref_positions = _parse_positions(structure_varray)

        wavenumbers = _parse_eigenvalues(root)
        eigenvectors = _parse_eigenvectors(root)
        displacements = eigenvectors / np.sqrt(atomic_weights)[:, np.newaxis]

        return Phonons(ref_positions, wavenumbers, displacements)
