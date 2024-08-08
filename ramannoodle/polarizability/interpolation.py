"""Polarizability models."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline, BSpline

from . import polarizability_utils
from . import PolarizabilityModel
from ..symmetry.symmetry_utils import (
    is_orthogonal_to_all,
    calculate_displacement,
    is_collinear_with_all,
)
from ..symmetry import StructuralSymmetry
from ..exceptions import InvalidDOFException, get_type_error

from .. import io
from ..io.io_utils import pathify_as_list
from ..exceptions import verify_ndarray_shape


def get_amplitude(
    cartesian_basis_vector: NDArray[np.float64],
    cartesian_displacement: NDArray[np.float64],
) -> float:
    """Get amplitude of a displacement in angstroms."""
    return float(
        np.dot(cartesian_basis_vector.flatten(), cartesian_displacement.flatten())
    )


class InterpolationModel(PolarizabilityModel):
    """Polarizability model based on interpolation around degrees of freedom.

    One is free to specify the interpolation order as well as the precise
    the degrees of freedom, so long as they are orthogonal. For example, one can
    employ first-order (linear) interpolation around phonon displacements to calculate
    a conventional Raman spectrum. One can achieve identical results -- often with fewer
    calculations -- by using first-order interpolations around atomic displacements.

    This model's key assumption is that each degree of freedom in a system modulates
    the polarizability **independently**.

    Parameters
    ----------
    structural_symmetry
    equilibrium_polarizability
        2D array with shape (3,3) giving polarizability of system at equilibrium. This
        would usually correspond to the minimum energy structure.

    """

    def __init__(
        self,
        structural_symmetry: StructuralSymmetry,
        equilibrium_polarizability: NDArray[np.float64],
    ) -> None:
        verify_ndarray_shape(
            "equilibrium_polarizability", equilibrium_polarizability, (3, 3)
        )
        self._structural_symmetry = structural_symmetry
        self._equilibrium_polarizability = equilibrium_polarizability
        self._cartesian_basis_vectors: list[NDArray[np.float64]] = []
        self._interpolations: list[BSpline] = []

    def get_polarizability(
        self, cartesian_displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return an estimated polarizability for a given cartesian displacement.

        Parameters
        ----------
        cartesian_displacement
            2D array with shape (N,3) where N is the number of atoms

        Returns
        -------
        :
            2D array with shape (3,3)

        """
        delta_polarizability: NDArray[np.float64] = np.zeros((3, 3))
        for basis_vector, interpolation in zip(
            self._cartesian_basis_vectors, self._interpolations
        ):
            try:
                amplitude = np.dot(
                    basis_vector.flatten(), cartesian_displacement.flatten()
                )
            except AttributeError as exc:
                raise TypeError("cartesian_displacement is not an ndarray") from exc
            except ValueError as exc:
                raise ValueError(
                    "cartesian_displacement has incompatible length "
                    f"({len(cartesian_displacement)}!={len(basis_vector)})"
                ) from exc
            delta_polarizability += interpolation(amplitude)

        return delta_polarizability + self._equilibrium_polarizability

    def add_dof(  # pylint: disable=too-many-locals
        self,
        displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        interpolation_order: int,
    ) -> None:
        """Add a degree of freedom (DOF).

        Specification of a DOF requires a displacement (how the atoms move) alongside
        displacement amplitudes and corresponding known polarizabilities for each
        amplitude. Alongside the DOF specified, all DOFs related by the system's
        symmetry will be added as well. The interpolation order can be specified,
        though one must ensure that sufficient data is available.

        Parameters
        ----------
        displacement
            2D array with shape (N,3) where N is the number of atoms. Units
            are arbitrary. Must be orthogonal to all previously added DOFs.
        amplitudes
            1D array of length L containing amplitudes in angstroms. Duplicate
            amplitudes are not allowed, including symmetrically equivalent
            amplitudes.
        polarizabilities
            3D array with shape (L,3,3) containing known polarizabilities for
            each amplitude.
        interpolation_order
            Must be less than the number of total number of amplitudes after
            symmetry considerations.

        Raises
        ------
        InvalidDOFException
            Provided degree of freedom was invalid.

        """
        try:
            parent_displacement = displacement / (np.linalg.norm(displacement) * 10)
        except TypeError as exc:
            raise TypeError("displacement is not an ndarray") from exc
        verify_ndarray_shape("amplitudes", amplitudes, (None,))
        verify_ndarray_shape(
            "polarizabilities", polarizabilities, (len(amplitudes), 3, 3)
        )

        parent_cartesian_basis_vector = (
            self._structural_symmetry.get_cartesian_displacement(parent_displacement)
        )
        # Check that the parent displacement is orthogonal to existing basis vectors
        result = is_orthogonal_to_all(
            parent_cartesian_basis_vector, self._cartesian_basis_vectors
        )
        if result != -1:
            raise InvalidDOFException(
                f"new dof is not orthogonal with existing dof (index={result})"
            )

        displacements_and_transformations = (
            self._structural_symmetry.get_equivalent_displacements(parent_displacement)
        )

        basis_vectors_to_add: list[NDArray[np.float64]] = []
        interpolations_to_add: list[BSpline] = []
        for dof_dictionary in displacements_and_transformations:
            child_displacement = dof_dictionary["displacements"][0]

            interpolation_x = [0.0]
            interpolation_y = [np.zeros((3, 3))]
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
                    delta_polarizability = (
                        polarizability - self._equilibrium_polarizability
                    )

                    interpolation_y.append(
                        (np.linalg.inv(rotation) @ delta_polarizability @ rotation)
                    )
            interpolation_x = np.array(interpolation_x)
            # If duplicate amplitudes are generated, too much data has
            # been provided
            duplicate = polarizability_utils.find_duplicates(interpolation_x)
            if duplicate is not None:
                raise InvalidDOFException(
                    f"due to symmetry, amplitude {duplicate} should not be specified"
                )

            if len(interpolation_x) <= interpolation_order:
                raise InvalidDOFException(
                    f"insufficient points ({len(interpolation_x)}) available for "
                    f"{interpolation_order}-order interpolation"
                )

            child_cartesian_basis_vector = (
                self._structural_symmetry.get_cartesian_displacement(child_displacement)
            )
            child_cartesian_basis_vector /= np.linalg.norm(child_cartesian_basis_vector)
            basis_vectors_to_add.append(child_cartesian_basis_vector)

            sort_indices = np.argsort(interpolation_x)
            try:
                interpolations_to_add.append(
                    make_interp_spline(
                        x=np.array(interpolation_x)[sort_indices],
                        y=np.array(interpolation_y)[sort_indices],
                        k=interpolation_order,
                        bc_type=None,
                    )
                )
            except ValueError as exc:
                if "non-negative k" in str(exc):
                    raise ValueError(
                        f"invalid interpolation_order: {interpolation_order} < 1"
                    ) from exc
                raise exc
            except TypeError as exc:
                raise get_type_error(
                    "interpolation_order", interpolation_order, "int"
                ) from exc

        self._cartesian_basis_vectors += basis_vectors_to_add
        self._interpolations += interpolations_to_add

    def add_dof_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
        interpolation_order: int,
    ) -> None:
        """Add a degree of freedom (DOF) from file(s).

        Required displacements, amplitudes, and polarizabilities are automatically
        determined from provided files. See "add_dof" for restrictions on these
        parameters.

        Parameters
        ----------
        filepaths
        file_format
            supports: "outcar"

        Raises
        ------
        FileNotFoundError
            File could not be found.
        InvalidDOFException
            DOF assembled from supplied files was invalid (see get_dof)

        """
        # Extract displacements, polarizabilities, and basis vector
        displacements = []
        polarizabilities = []
        filepaths = pathify_as_list(filepaths)
        for filepath in filepaths:
            fractional_positions, polarizability = io.read_positions_and_polarizability(
                filepath, file_format
            )
            try:
                displacement = calculate_displacement(
                    fractional_positions,
                    self._structural_symmetry.get_fractional_positions(),
                )
            except ValueError as exc:
                raise InvalidDOFException(f"incompatible outcar: {filepath}") from exc
            displacements.append(displacement)
            polarizabilities.append(polarizability)
        result = is_collinear_with_all(displacements[0], displacements)
        if result != -1:
            raise InvalidDOFException(
                f"displacement (file-index={result}) is not collinear"
            )
        cartesian_basis_vector = self._structural_symmetry.get_cartesian_displacement(
            displacements[0]
        )
        cartesian_basis_vector /= np.linalg.norm(cartesian_basis_vector)

        # Calculate amplitudes
        amplitudes = []
        for displacement in displacements:
            cartesian_displacement = (
                self._structural_symmetry.get_cartesian_displacement(displacement)
            )
            amplitudes.append(
                get_amplitude(cartesian_basis_vector, cartesian_displacement)
            )

        self.add_dof(
            displacements[0],
            np.array(amplitudes),
            np.array(polarizabilities),
            interpolation_order,
        )
