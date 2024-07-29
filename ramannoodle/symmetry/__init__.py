"""Symmetry for structures, Raman peaks, etc."""

import numpy as np
from numpy.typing import NDArray
import spglib

from . import symmetry_utils
from ..exceptions import SymmetryException


class StructuralSymmetry:
    """Symmetries of a crystal structure."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        atomic_numbers: NDArray[np.int32],
        lattice: NDArray[np.float64],
        fractional_positions: NDArray[np.float64],
        symprec: float = 1e-5,
        angle_tolerance: float = -1.0,
    ) -> None:
        self._atomic_numbers = atomic_numbers
        self._lattice = lattice
        self._fractional_positions = fractional_positions

        cell = (list(lattice), list(fractional_positions), list(atomic_numbers))
        self._symmetry_dict: dict[str, NDArray[np.float64]] | None = (
            spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        )
        if self._symmetry_dict is None:
            raise SymmetryException("symmetry search failed")

        self._rotations = self._symmetry_dict["rotations"]
        self._translations = self._symmetry_dict["translations"]
        self._permutation_matrices = symmetry_utils.compute_permutation_matrices(
            self._rotations, self._translations, self._fractional_positions
        )

    def get_num_nonequivalent_atoms(self) -> int:
        """Returns the number of nonequivalent atoms."""
        assert self._symmetry_dict is not None
        return len(set(self._symmetry_dict["equivalent_atoms"]))

    def get_equivalent_displacements(
        self, displacement: NDArray[np.float64]
    ) -> list[dict[str, list[NDArray[np.float64]]]]:
        """Calculates and returns all symmetrically equivalent displacements.
        The return is a little complicated.
        [('displacements' : [...], {'transformations' : [...]}]
        We want to guarantee that all displacements are orthogonal between
        the dictionaries and are all collinear and unique within the
        dictionaries.
        """

        assert (displacement >= -0.5).all() and (displacement <= 0.5).all()

        ref_positions = symmetry_utils.add_fractional_positions(
            self._fractional_positions, displacement
        )

        result = []
        orthogonal_displacements: list[NDArray[np.float64]] = []
        for rotation, translation, permutation_matrix in zip(
            self._rotations, self._translations, self._permutation_matrices
        ):

            # Transform, permute, then get displacements
            candidate_positions = symmetry_utils.transform_fractional_positions(
                ref_positions, rotation, translation
            )
            candidate_positions = permutation_matrix @ candidate_positions
            candidate_displacement = symmetry_utils.calculate_displacement(
                candidate_positions, self._fractional_positions
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
