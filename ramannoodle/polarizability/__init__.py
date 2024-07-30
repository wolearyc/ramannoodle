"""Polarizability models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline, BSpline

from . import polarizability_utils
from ..symmetry.symmetry_utils import is_orthogonal_to_all
from ..symmetry import StructuralSymmetry
from ..exceptions import InvalidDOFException


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract polarizability model."""

    @abstractmethod
    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return an estimated polarizability for a given displacement."""


class InterpolationPolarizabilityModel(PolarizabilityModel):
    """Polarizability model based on interpolation around degrees of freedom.

    One is free to specify the interpolation order as well as the precise
    form of the degrees of freedom, so long as they are orthogonal. For example, one can
    employ first-order (linear) interpolation around phonon displacements to calculate
    a conventional Raman spectrum. One can achieve similar results with fewer
    calculations by using first-order interpolations around atomic displacements.
    """

    def __init__(
        self,
        structural_symmetry: StructuralSymmetry,
        equilibrium_polarizability: NDArray[np.float64],
    ) -> None:
        """Construct model.

        Parameters
        ----------
        structural_symmetry: StructuralSymmetry
        equilibrium_polarizability: NDArray[np.float64]
            Polarizability (3x3) of system at "equilibrium", i.e., minimized structure.

            Raman spectra calculated using this model do not explicitly depend on this
            value. However, specifying the actual value is recommended in order to
            compute the correct polarizability magnitude.
        """
        self._structural_symmetry = structural_symmetry
        self._equilibrium_polarizability = equilibrium_polarizability
        self._basis_vectors: list[NDArray[np.float64]] = []
        self._interpolations: list[BSpline] = []

    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return an estimated polarizability for a given displacement."""
        polarizability: NDArray[np.float64] = np.zeros((3, 3))
        for basis_vector, interpolation in zip(
            self._basis_vectors, self._interpolations
        ):
            projected_displacement = (
                self._structural_symmetry.get_cartesian_displacement(
                    np.dot(basis_vector.flatten(), displacement.flatten())
                    * basis_vector
                )
            )
            amplitude = np.linalg.norm(projected_displacement)
            polarizability += interpolation(amplitude)

        return polarizability + self._equilibrium_polarizability

    def add_dof(  # pylint: disable=too-many-locals
        self,
        displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        interpolation_dim: int,
    ) -> None:
        """Add a degree of freedom."""
        parent_displacement = displacement / (np.linalg.norm(displacement) * 10)

        # Check that the parent displacement is orthogonal to existing basis vectors
        result = is_orthogonal_to_all(parent_displacement, self._basis_vectors)
        if result != -1:
            raise InvalidDOFException(
                f"new dof is not orthogonal with existing dof (index={result})"
            )
        if len(amplitudes) == 0:
            raise ValueError("no amplitudes provided")
        if len(amplitudes) != len(polarizabilities):
            raise ValueError(
                f"unequal numbers of amplitudes ({len(amplitudes)})) and "
                f"polarizabilities ({len(polarizabilities)})"
            )

        displacements_and_transformations = (
            self._structural_symmetry.get_equivalent_displacements(parent_displacement)
        )

        basis_vectors_to_add: list[NDArray[np.float64]] = []
        interpolations_to_add: list[BSpline] = []
        for dof_dictionary in displacements_and_transformations:
            child_displacement = dof_dictionary["displacements"][0]

            interpolation_x = []
            interpolation_y = []
            for collinear_displacement, transformation in zip(
                dof_dictionary["displacements"], dof_dictionary["transformations"]
            ):
                _index = np.unravel_index(
                    np.argmax(np.abs(child_displacement)), child_displacement.shape
                )
                multiplier = child_displacement[_index] / collinear_displacement[_index]

                for amplitude, polarizability in zip(amplitudes, polarizabilities):
                    interpolation_x.append(multiplier * amplitude)
                    rotation = transformation[0]
                    interpolation_y.append(
                        (rotation @ polarizability @ np.linalg.inv(rotation))
                        - self._equilibrium_polarizability
                    )

            # If duplicate amplitudes are generated, too much data has
            # been provided
            duplicate = polarizability_utils.find_duplicates(interpolation_x)
            if duplicate is not None:
                raise InvalidDOFException(
                    f"due to symmetry, amplitude {duplicate} should not be specified"
                )

            if len(interpolation_x) <= interpolation_dim:
                raise InvalidDOFException(
                    f"insufficient amplitudes ({len(interpolation_x)}) available for"
                    f"{interpolation_dim}-dimensional interpolation"
                )
            assert len(interpolation_x) > 1
            basis_vectors_to_add.append(
                child_displacement / np.linalg.norm(child_displacement)
            )
            sort_indices = np.argsort(interpolation_x)
            interpolations_to_add.append(
                make_interp_spline(
                    x=np.array(interpolation_x)[sort_indices],
                    y=np.array(interpolation_y)[sort_indices],
                    k=interpolation_dim,
                    bc_type=None,
                )
            )

        self._basis_vectors += basis_vectors_to_add
        self._interpolations += interpolations_to_add
