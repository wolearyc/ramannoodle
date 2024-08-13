"""Reference structure."""

import numpy as np
from numpy.typing import NDArray
import spglib

from ramannoodle.exceptions import (
    SymmetryException,
    get_type_error,
    verify_ndarray_shape,
    verify_list_len,
)
from ramannoodle.structure.structure_utils import (
    displace_positions,
    transform_positions,
    apply_pbc,
)
from ramannoodle.globals import ATOM_SYMBOLS
from ramannoodle.structure import symmetry_utils
import ramannoodle.structure.structure_utils


def _compute_permutation_matrices(
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expresses a series of rotation/translations as permutation matrices."""
    # Ensure no atom is at unit cell boundary by shifting
    # center of mass
    center_of_mass_shift = np.array([0.5, 0.5, 0.5]) - np.mean(positions, axis=0)

    permutation_matrices = []
    for rotation, translation in zip(rotations, translations):
        permutation_matrices.append(
            _get_positions_permutation_matrix(
                displace_positions(positions, center_of_mass_shift),
                transform_positions(
                    positions, rotation, translation + center_of_mass_shift
                ),
            )
        )
    return np.array(permutation_matrices)


def _get_positions_permutation_matrix(
    reference_positions: NDArray[np.float64], permuted_positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Calculate a permutation matrix between reference and permuted positions.

    .. warning::
        Arguments must be true permutations of each other. This function does not
        correct for periodic boundary conditions, so it needs to be supplied a
        structure without atoms at the unit cell boundaries.

    Parameters
    ----------
    reference_positions
        A 2D array with shape (N,3)
    permuted_positions
        A 2D array with shape (N,3).

    """
    reference_positions = apply_pbc(reference_positions)
    permuted_positions = apply_pbc(permuted_positions)

    argsort_reference = np.lexsort(
        (
            reference_positions[:, 2],
            reference_positions[:, 1],
            reference_positions[:, 0],
        )
    )
    argsort_permuted = np.lexsort(
        (permuted_positions[:, 2], permuted_positions[:, 1], permuted_positions[:, 0])
    )
    sorted_reference = reference_positions[argsort_reference]
    sorted_permuted = permuted_positions[argsort_permuted]
    if not np.isclose(sorted_reference, sorted_permuted).all():
        raise ValueError("permuted is not a permutation of reference")

    permutation_matrix = np.zeros((len(reference_positions), len(reference_positions)))
    permutation_matrix[tuple(argsort_reference), tuple(argsort_permuted)] = 1

    return permutation_matrix


class ReferenceStructure:
    """Crystal structure, typically used as a reference in polarizability models.

    Parameters
    ----------
    atomic_numbers
        1D list of length N where N is the number of atoms.
    lattice
        Lattice vectors expressed as a 2D array with shape (3,3).
    positions
        2D array with shape (N,3) where N is the number of atoms
    symprec
        Symmetry precision parameter for spglib.
    angle_tolerance
        Symmetry precision parameter for spglib.

    Raises
    ------
    SymmetryException
        Symmetry could not be determined for supplied structure.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        atomic_numbers: list[int],
        lattice: NDArray[np.float64],
        positions: NDArray[np.float64],
        symprec: float = 1e-5,
        angle_tolerance: float = -1.0,
    ) -> None:
        verify_list_len("atomic_numbers", atomic_numbers, None)
        verify_ndarray_shape("lattice", lattice, (3, 3))
        verify_ndarray_shape("positions", positions, (len(atomic_numbers), 3))

        self._atomic_numbers = atomic_numbers
        self._lattice = lattice
        self._positions = positions

        cell = (list(lattice), list(positions), atomic_numbers)
        self._symmetry_dict: dict[str, NDArray[np.float64]] | None = (
            spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        )
        if self._symmetry_dict is None:
            raise SymmetryException("Symmetry search failed. Check structure.")

        self._rotations = self._symmetry_dict["rotations"]
        self._translations = self._symmetry_dict["translations"]
        self._permutation_matrices = _compute_permutation_matrices(
            self._rotations, self._translations, self._positions
        )

    @property
    def atomic_numbers(self) -> list[int]:
        """Return atomic numbers."""
        return self._atomic_numbers

    @property
    def lattice(self) -> NDArray[np.float64]:
        """Return lattice."""
        return self._lattice

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return fractional positions."""
        return self._positions

    def get_num_nonequivalent_atoms(self) -> int:
        """Return number of nonequivalent atoms."""
        assert self._symmetry_dict is not None
        return len(set(self._symmetry_dict["equivalent_atoms"]))

    def get_equivalent_atom_dict(self) -> dict[int, list[int]]:
        """Get dictionary of equivalent atoms."""
        assert self._symmetry_dict is not None
        result: dict[int, list[int]] = {}
        for index, equiv_index in enumerate(self._symmetry_dict["equivalent_atoms"]):
            if index == equiv_index:
                result[index] = []
            else:
                result[equiv_index].append(index)
        return result

    def get_equivalent_displacements(
        self, displacement: NDArray[np.float64]
    ) -> list[dict[str, list[NDArray[np.float64]]]]:
        """Calculate symmetrically equivalent displacements.

        Parameters
        ----------
        displacement
            2D array with shape (N,3) where N is the number of atoms.

        Returns
        -------
        :
            List of dictionaries containing displacements and transformations,
            accessed using the 'displacements' and 'transformations' keys. Displacements
            within each dictionary will be collinear, corresponding to
            the same degree of freedom. The provided transformations are those that
            transform the parameter `displacements` into that degree of freedom.

        """
        displacement = ramannoodle.structure.structure_utils.apply_pbc_displacement(
            displacement
        )
        # Scale the displacement for numerical reasons.
        displacement = displacement / (np.linalg.norm(displacement) * 10)

        ref_positions = ramannoodle.structure.structure_utils.displace_positions(
            self._positions, displacement
        )

        result = []
        orthogonal_displacements: list[NDArray[np.float64]] = []
        for rotation, translation, permutation_matrix in zip(
            self._rotations, self._translations, self._permutation_matrices
        ):

            # Transform, permute, then get candidate displacement
            candidate_positions = (
                ramannoodle.structure.structure_utils.transform_positions(
                    ref_positions, rotation, translation
                )
            )
            candidate_positions = permutation_matrix @ candidate_positions
            candidate_displacement = (
                ramannoodle.structure.structure_utils.calculate_displacement(
                    candidate_positions, self._positions
                )
            )

            orthogonal_result = symmetry_utils.is_orthogonal_to_all(
                candidate_displacement.flatten(),
                [item.flatten() for item in orthogonal_displacements],
            )

            # If new orthogonal displacement discovered
            if orthogonal_result == -1:
                result.append(
                    {
                        "displacements": [candidate_displacement],
                        "transformations": [(rotation, translation)],
                    }
                )
                orthogonal_displacements.append(candidate_displacement)
            else:
                # Candidate can be collinear to maximum of one orthogonal vector
                collinear_result = symmetry_utils.is_non_collinear_with_all(
                    candidate_displacement,
                    orthogonal_displacements,
                )
                if collinear_result != -1:  # collinear
                    collinear_displacements = result[collinear_result]["displacements"]

                    # Check if we have a duplicate
                    is_duplicate = False
                    for collinear_displacement in collinear_displacements:
                        if np.isclose(
                            candidate_displacement, collinear_displacement, atol=1e-05
                        ).all():
                            is_duplicate = True
                            break
                    # If unique, add
                    if not is_duplicate:
                        result[collinear_result]["displacements"].append(
                            candidate_displacement
                        )
                        result[collinear_result]["transformations"].append(
                            (rotation, translation)
                        )

        return result

    def get_cartesian_displacement(
        self, fractional_displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a fractional displacement into cartesian coordinates.

        Parameters
        ----------
        fractional_displacement
            2D array with shape (N,3) where N is the number of atoms
        """
        fractional_displacement = (
            ramannoodle.structure.structure_utils.apply_pbc_displacement(
                fractional_displacement
            )
        )

        return fractional_displacement @ self._lattice

    def get_atom_indexes(self, atom_symbols: str | list[str]) -> list[int]:
        """Return atom indexes with matching symbols.

        Parameters
        ----------
        atom_symbols
            If integer or list of integers, specifies atom indexes. If string or list
            of strings, specifies atom symbols. Mixtures of integers and strings are
            allowed.
        """
        symbols = [ATOM_SYMBOLS[number] for number in self._atomic_numbers]
        indexes = []
        if isinstance(atom_symbols, str):
            atom_symbols = [atom_symbols]
        try:
            for index, symbol in enumerate(symbols):
                if symbol in atom_symbols:
                    indexes.append(index)
        except TypeError as err:
            raise get_type_error("atom_symbols", atom_symbols, "list") from err
        return indexes