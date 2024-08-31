"""Reference structure."""

import copy

import numpy as np
from numpy.typing import NDArray
import spglib

from ramannoodle.exceptions import (
    SymmetryException,
    get_type_error,
    verify_ndarray_shape,
    verify_list_len,
    get_shape_error,
)
from ramannoodle.structure.structure_utils import (
    displace_positions,
    transform_positions,
    apply_pbc_displacement,
    calc_displacement,
)
from ramannoodle.globals import ATOM_SYMBOLS
from ramannoodle.structure import symmetry_utils


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

    Parameters
    ----------
    reference_positions
        A 2D array with shape (N,3)
    permuted_positions
        A 2D array with shape (N,3).

    """
    # Compute pairwise distance matrix.
    reference_positions = np.expand_dims(reference_positions, 0)
    permuted_positions = np.expand_dims(permuted_positions, 1)
    displacement = reference_positions - permuted_positions
    displacement = np.where(
        displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1
    )
    distance_matrix = np.sqrt(np.sum(displacement**2, axis=-1))
    permutation_matrix = (distance_matrix < 1e-5).T

    return permutation_matrix


class ReferenceStructure:
    """Reference crystal structure, typically used by polarizability models.

    Parameters
    ----------
    atomic_numbers
        | List of length N where N is the number of atoms.
    lattice
        | (Å) 2D array with shape (3,3).
    positions
        | (fractional) 2D array with shape (N,3).
    symprec
        | (Å) Distance tolerance for symmetry search (spglib).
    angle_tolerance
        | (°) Angle tolerance for symmetry search (spglib).

    Raises
    ------
    SymmetryException
        Structural symmetry determination failed.

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
        """Get (a copy of) atomic numbers."""
        return copy.copy(self._atomic_numbers)

    @property
    def num_atoms(self) -> int:
        """Get number of atoms."""
        return len(self._atomic_numbers)

    @property
    def lattice(self) -> NDArray[np.float64]:
        """Get (a copy of) lattice.

        Returns
        -------
        :
            Å | 2D array with shape (3,3).
        """
        return self._lattice.copy()

    @property
    def positions(self) -> NDArray[np.float64]:
        """Get (a copy of) fractional positions.

        Returns
        -------
        :
            (fractional) 2D array with shape (N,3) where N is the number of atoms.
        """
        return self._positions.copy()

    @property
    def num_nonequivalent_atoms(self) -> int:
        """Get number of nonequivalent atoms."""
        assert self._symmetry_dict is not None
        return len(set(self._symmetry_dict["equivalent_atoms"]))

    def get_equivalent_atom_dict(self) -> dict[int, list[int]]:
        """Get dictionary of equivalent atoms indexes.

        Returns
        -------
        :
            dict:
                | atom index --> list of equivalent atom indexes

        """
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
            | (fractional) 2D array with shape (N,3) where N is the number of atoms.

        Returns
        -------
        :
            List of dictionaries containing displacements and transformations,
            accessed using the 'displacements' and 'transformations' keys. Displacements
            within each dictionary will be collinear, corresponding to
            the same degree of freedom. The provided transformations are those that
            transform ``displacement`` into that degree of freedom. Displacements are
            in fractional coordinates.

        """
        displacement = apply_pbc_displacement(displacement)
        # Scale the displacement for numerical reasons.
        displacement = displacement / (np.linalg.norm(displacement) * 10)

        ref_positions = displace_positions(self._positions, displacement)

        result = []
        orthogonal_displacements: list[NDArray[np.float64]] = []
        for rotation, translation, permutation_matrix in zip(
            self._rotations, self._translations, self._permutation_matrices
        ):

            # Transform, permute, then get candidate displacement
            candidate_positions = transform_positions(
                ref_positions, rotation, translation
            )
            candidate_positions = permutation_matrix @ candidate_positions
            candidate_displacement = calc_displacement(
                self._positions,
                candidate_positions,
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

    def get_cart_displacement(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a (fractional) displacement into Cartesian coordinates.

        Parameters
        ----------
        displacement
            | (fractional) Array with shape (...,N,3)  where N is the number of atoms.

        Returns
        -------
        :
            (Å) | Array with shape (...,N,3).
        """
        displacement = apply_pbc_displacement(displacement)

        return displacement @ self._lattice

    def get_cart_direction(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert a (fractional) direction into cartesian coordinates.

        Parameters
        ----------
        direction
            | (fractional) 1D array with shape (3,).

        Returns
        -------
        :
            (Å) 1D array with shape (3,).
        """
        direction = apply_pbc_displacement(direction)
        try:
            return np.array([direction]) @ self._lattice
        except ValueError as exc:
            raise get_shape_error("direction", direction, "(3,)") from exc

    def get_frac_displacement(
        self, cart_displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a Cartesian displacement into fractional coordinates.

        Parameters
        ----------
        cart_displacement
            | (Å) 2D array with shape (N,3) where N is the number of atoms.

        Returns
        -------
        :
            (fractional) 2D array with shape (N,3).
        """
        verify_ndarray_shape("cart_displacement", cart_displacement, (None, 3))
        displacement = (cart_displacement) @ np.linalg.inv(self.lattice)
        return apply_pbc_displacement(displacement)

    def get_frac_direction(
        self, cart_direction: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a Cartesian direction into fractional coordinates.

        Parameters
        ----------
        cart_direction
            | (Å) 1D array with shape (3,).

        Returns
        -------
        :
            | (fractional) 1D array with shape (3,).
        """
        verify_ndarray_shape("direction", cart_direction, (3,))
        displacement = np.array([cart_direction]) @ np.linalg.inv(self.lattice)
        return apply_pbc_displacement(displacement[0])

    def get_atom_indexes(self, atom_symbols: str | list[str]) -> list[int]:
        """Return atom indexes with matching symbols.

        Parameters
        ----------
        atom_symbols
            If integer or list of integers, specifies atom indexes. If string or list
            of strings, specifies atom symbols.

            Mixtures of indexes and symbols are allowed.
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
