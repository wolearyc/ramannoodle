"""Parent class for polarizability models."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline, BSpline
import spglib

from . import polarizability_utils
from ..exceptions import InvalidDOFException, SymmetryException


class StructuralSymmetry:
    """Represents symmetries of a crystal structure."""

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
        self._permutation_matrices = polarizability_utils.compute_permutation_matrices(
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

        ref_positions = polarizability_utils.add_fractional_positions(
            self._fractional_positions, displacement
        )

        result = []
        orthogonal_displacements: list[NDArray[np.float64]] = []
        for rotation, translation, permutation_matrix in zip(
            self._rotations, self._translations, self._permutation_matrices
        ):

            # Transform, permute, then get displacements
            candidate_positions = polarizability_utils.transform_fractional_positions(
                ref_positions, rotation, translation
            )
            candidate_positions = permutation_matrix @ candidate_positions
            candidate_displacement = polarizability_utils.subtract_fractional_positions(
                candidate_positions, self._fractional_positions
            )

            orthogonal_result = polarizability_utils.is_orthogonal_to_all(
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
                collinear_result = polarizability_utils.is_non_collinear_with_all(
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


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Represents a polarizability model"""

    @abstractmethod
    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Returns a polarizability for a set of atomic displacements."""


class InterpolationPolarizabilityModel(PolarizabilityModel):
    """This model uses interpolation around independent degrees of
    freedom to estimate polarizabilities.
    With linear interpolation + phonon displacements, we get a standard
    raman-tensor-based spectrum.
    With linear interpolation + site displacements, we get an atomic raman
    tensor-based spectrum. This class also needs to take careful care
    to obey symmetry operations.

    """

    def __init__(
        self, structural_symmetry: StructuralSymmetry
    ) -> None:  # - Generate all symmetry equivalent displacements.
        self._structural_symmetry = structural_symmetry
        self._dof_displacements: list[NDArray[np.float64]] = []
        self._dof_interpolations: list[BSpline] = []

    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Returns a polarizability for a given displacement."""

        # Project displacement onto each dof_displacement, use the coordinate to
        # define the interpolation.

        return np.array([])

    def add_dof(
        self,
        displacements_and_polarizabilities: list[
            tuple[NDArray[np.float64], NDArray[np.float64]]
        ],
        interpolation_dim: int,
    ) -> None:
        """Adds a degree of freedom as well as any symmetrically
        equivalent degrees of freedom."""

        if len(displacements_and_polarizabilities) == 0:
            raise InvalidDOFException("no dof provided")

        dof_displacement, _ = displacements_and_polarizabilities[0]
        dof_displacement /= np.linalg.norm(dof_displacement)

        # Check that all provided displacements are collinear
        for i, (displacement, _) in enumerate(displacements_and_polarizabilities):
            if not polarizability_utils.are_collinear(dof_displacement, displacement):
                raise InvalidDOFException(f"displacement (index={i}) is not collinear")

        # Check that new displacement is orthogonal with existing displacements
        result = polarizability_utils.is_orthogonal_to_all(
            dof_displacement, self._dof_displacements
        )
        if result != -1:
            raise InvalidDOFException(
                f"new dof is not orthogonal with existing dof (index={result})"
            )

        # Get symmetrically equivalent displacements/transformations
        self._structural_symmetry.get_equivalent_displacements(dof_displacement)

        # Extract any collinear displacements it finds

        # Double check that remaining displacements are orthogonal

        interpolation = make_interp_spline(
            x=[d for d, _ in displacements_and_polarizabilities],
            y=[p for _, p in displacements_and_polarizabilities],
            k=interpolation_dim,
            bc_type="natural",
        )

        # new_dof_displacements, new_dof_interpolations =

        self._dof_displacements.append(dof_displacement)
        self._dof_interpolations.append(interpolation)


#       # Go through all transformations and fill in for all equivalent atoms
#
