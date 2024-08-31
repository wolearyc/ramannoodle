"""Polarizability model based on interpolation around degrees of freedom."""

# This is not ideal, but is required for Python 3.10 support.
# In future versions, we can use "from typing import Self"
from __future__ import annotations

from pathlib import Path
import copy
from typing import Iterable
from warnings import warn
import itertools

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.interpolate import make_interp_spline, BSpline

from ramannoodle.globals import ANSICOLORS
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.structure.structure_utils import calc_displacement
from ramannoodle.structure.symmetry_utils import (
    is_orthogonal_to_all,
    is_collinear_with_all,
)
from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.exceptions import (
    InvalidDOFException,
    get_type_error,
    get_shape_error,
    verify_ndarray_shape,
    DOFWarning,
    UsageError,
)
import ramannoodle.io.generic as generic_io
from ramannoodle.io.io_utils import pathify_as_list


def get_amplitude(
    cart_basis_vector: NDArray[np.float64],
    cart_displacement: NDArray[np.float64],
) -> float:
    """Get amplitude of a displacement.

    Returns
    -------
    :
        (Å)
    """
    return float(np.dot(cart_basis_vector.flatten(), cart_displacement.flatten()))


def find_duplicates(vectors: Iterable[ArrayLike]) -> NDArray | None:
    """Return duplicate vector in a list or None if no duplicates found.

    :meta private:
    """
    try:
        combinations = itertools.combinations(vectors, 2)
    except TypeError as exc:
        raise get_type_error("vectors", vectors, "Iterable") from exc
    try:
        for vector_1, vector_2 in combinations:
            if np.isclose(vector_1, vector_2).all():
                return np.array(vector_1)
        return None
    except TypeError as exc:
        raise TypeError("elements of vectors are not array_like") from exc


class InterpolationModel(PolarizabilityModel):
    """Polarizability model based on interpolation around degrees of freedom (DOFs).

    One is free to specify the interpolation order as well as the precise
    the DOFs, so long as they are orthogonal. For example, one can
    employ first-order (linear) interpolation around phonon displacements to calculate
    a conventional Raman spectrum. One can achieve identical results -- often with fewer
    calculations -- by using first-order interpolations around atomic displacements.

    This model's key assumption is that each degree of freedom in a system modulates
    the polarizability **independently**.

    Parameters
    ----------
    ref_structure
        | Reference structure on which to base the model.
    ref_polarizability
        | 2D array with shape (3,3) with polarizability of the reference structure.
    is_dummy_model

    """

    def __init__(
        self,
        ref_structure: ReferenceStructure,
        ref_polarizability: NDArray[np.float64],
        is_dummy_model: bool = False,
    ) -> None:
        if is_dummy_model:
            ref_polarizability = np.zeros((3, 3))
        verify_ndarray_shape("ref_polarizability", ref_polarizability, (3, 3))

        self._ref_structure = ref_structure
        self._ref_polarizability = ref_polarizability
        self._is_dummy_model = is_dummy_model
        self._cart_basis_vectors: list[NDArray[np.float64]] = []
        self._interpolations: list[BSpline] = []
        self._mask: NDArray[np.bool] = np.array([], dtype="bool")

    @property
    def ref_structure(self) -> ReferenceStructure:
        """Get (a copy of) reference structure."""
        return copy.deepcopy(self._ref_structure)

    @property
    def ref_polarizability(self) -> NDArray[np.float64]:
        """Get (a copy of) reference polarizability.

        Returns
        -------
        :
            2D array with shape (3,3).
        """
        return self._ref_polarizability.copy()

    @property
    def is_dummy_model(self) -> bool:
        """Get whether model is a dummy model."""
        return self._is_dummy_model

    @property
    def cart_basis_vectors(self) -> list[NDArray[np.float64]]:
        """Get (a copy of) cartesian basis vectors.

        Returns
        -------
        :
            (Å) List of length J containing 2D arrays with shape (N,3) where J is the
            number of specified degrees of freedom and N is the number of atoms.

        """
        return copy.deepcopy(self._cart_basis_vectors)

    @property
    def interpolations(self) -> list[BSpline]:
        """Get (a copy of) interpolations.

        Returns
        -------
        :
            List of length J where J is the number of specified degrees of freedom.
        """
        return copy.deepcopy(self._interpolations)

    @property
    def mask(self) -> NDArray[np.bool]:
        """Get (a copy of) mask.

        Returns
        -------
        :
            1D array with shape (J,) where J is the number of specified degrees of
            freedom.
        """
        return self._mask.copy()

    @mask.setter
    def mask(self, value: NDArray[np.bool]) -> None:
        """Set mask.

        ..warning:: To avoid unintentional use of masked models, we discourage masking
                    in-place. Instead, consider using :meth:`get_masked_model`.

        Parameters
        ----------
        mask
            1D array of size (N,) where N is the number of specified degrees
            of freedom (DOFs).

            If an element is False, its corresponding DOF will be "masked" and excluded
            from polarizability calculations.
        """
        verify_ndarray_shape("mask", value, self._mask.shape)
        self._mask = value

    def calc_polarizabilities(
        self, positions_batch: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return estimated polarizabilities for a batch of fractional positions.

        Parameters
        ----------
        positions_batch
            | (fractional) 3D array with shape (S,N,3) where S is the number of samples
            | and N is the number of atoms.

        Returns
        -------
        :
            3D array with shape (S,3,3).

        Raises
        ------
        UsageError
            Model is a dummy model.

        """
        try:
            delta_polarizabilities: NDArray[np.float64] = np.zeros(
                (positions_batch.shape[0], 3, 3)
            )
            cart_displacements = self._ref_structure.get_cart_displacement(
                calc_displacement(self._ref_structure.positions, positions_batch)
            )
            cart_displacements = cart_displacements.reshape(
                cart_displacements.shape[0],
                cart_displacements.shape[1] * cart_displacements.shape[2],
            )
        except (AttributeError, TypeError) as exc:
            raise get_type_error("positions", positions_batch, "ndarray") from exc
        except (ValueError, IndexError) as exc:
            raise get_shape_error(
                "positions", positions_batch, f"(_,{self._ref_structure.num_atoms},3)"
            ) from exc

        try:

            for basis_vector, interpolation, mask in zip(
                self._cart_basis_vectors,
                self._interpolations,
                self._mask,
                strict=True,
            ):
                amplitudes = np.einsum(
                    "i,ji", basis_vector.flatten(), cart_displacements
                )
                delta_polarizabilities += (1 - mask) * np.array(
                    interpolation(amplitudes), dtype="float64"
                )
        except ValueError as err:
            if self._is_dummy_model:
                raise UsageError(
                    "dummy model cannot calculate polarizabilities"
                ) from err
            raise err

        return delta_polarizabilities + self._ref_polarizability

    def _get_dof(  # pylint: disable=too-many-locals
        self,
        parent_displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        include_ref_polarizability: bool,
    ) -> tuple[
        list[NDArray[np.float64]], list[list[float]], list[list[NDArray[np.float64]]]
    ]:
        """Calculate and return basis vectors and interpolation points for DOF(s).

        This method will check the displacement to make sure it's valid, but will not
        check the amplitudes. Amplitude checking is performed in
        ``_construct_and_add_interpolations``, as this method has access to the
        interpolation order.

        Parameters
        ----------
        parent_displacement
            Displacement of the parent DOF.
        amplitudes
            Amplitudes (of the parent DOF).
        polarizabilities
            Polarizabilities (of the parent DOF).

        Returns
        -------
        :
            3-tuple of the form (basis vectors, interpolation_xs, interpolation_ys)
        """
        # Check that the parent displacement is orthogonal to existing basis vectors
        parent_cart_basis_vector = self._ref_structure.get_cart_displacement(
            parent_displacement
        )
        result = is_orthogonal_to_all(
            parent_cart_basis_vector, self._cart_basis_vectors
        )
        if result != -1:
            raise InvalidDOFException(
                f"new dof is not orthogonal with existing dof (index={result})"
            )

        displacements_and_transformations = (
            self._ref_structure.get_equivalent_displacements(parent_displacement)
        )

        basis_vectors: list[NDArray[np.float64]] = []
        interpolation_xs: list[list[float]] = []
        interpolation_ys: list[list[NDArray[np.float64]]] = []
        for dof_dictionary in displacements_and_transformations:
            child_displacement = dof_dictionary["displacements"][0]

            interpolation_x: list[float] = []
            interpolation_y: list[NDArray[np.float64]] = []
            if include_ref_polarizability:
                interpolation_x.append(0.0)
                interpolation_y.append(np.zeros((3, 3)))

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
                    delta_polarizability = polarizability - self._ref_polarizability

                    interpolation_y.append(
                        (np.linalg.inv(rotation) @ delta_polarizability @ rotation)
                    )

            child_cart_basis_vector = self._ref_structure.get_cart_displacement(
                child_displacement
            )
            child_cart_basis_vector /= np.linalg.norm(child_cart_basis_vector)

            basis_vectors.append(child_cart_basis_vector)
            interpolation_xs.append(interpolation_x)
            interpolation_ys.append(interpolation_y)

        return (basis_vectors, interpolation_xs, interpolation_ys)

    def _construct_and_add_interpolations(
        self,
        basis_vectors_to_add: list[NDArray[np.float64]],
        interpolation_xs: list[list[float]],
        interpolation_ys: list[list[NDArray[np.float64]]],
        interpolation_order: int,
    ) -> None:
        """Construct interpolations and add them to the model.

        Raises
        ------
        InvalidDOFException
        """
        # make_interp_spline accepts k=0 (useless), so check here
        if not isinstance(interpolation_order, int):
            raise get_type_error("interpolation_order", interpolation_order, "int")
        if interpolation_order < 1:
            raise ValueError(f"invalid interpolation_order: {interpolation_order} < 1")

        interpolations_to_add: list[BSpline] = []
        for interpolation_x, interpolation_y in zip(interpolation_xs, interpolation_ys):

            # Duplicate amplitudes means too much data has been provided.
            duplicate = find_duplicates(interpolation_x)
            if duplicate is not None:
                raise InvalidDOFException(
                    f"due to symmetry, amplitude {duplicate} should not be specified"
                )

            if len(interpolation_x) <= interpolation_order:
                raise InvalidDOFException(
                    f"insufficient points ({len(interpolation_x)}) available for "
                    f"{interpolation_order}-order interpolation"
                )

            # Warn user if amplitudes don't span zero
            max_amplitude = np.max(interpolation_x)
            min_amplitude = np.min(interpolation_x)
            if np.isclose(max_amplitude, 0, atol=1e-3).all() or max_amplitude <= 0:
                warn(
                    "max amplitude <= 0, when usually it should be > 0",
                    DOFWarning,
                )
            if np.isclose(min_amplitude, 0, atol=1e-3).all() or min_amplitude >= 0:
                warn(
                    "min amplitude >= 0, when usually it should be < 0",
                    DOFWarning,
                )

            sort_indices = np.argsort(interpolation_x)
            interpolations_to_add.append(
                make_interp_spline(
                    x=np.array(interpolation_x)[sort_indices],
                    y=np.array(interpolation_y)[sort_indices],
                    k=interpolation_order,
                    bc_type=None,
                )
            )

        self._cart_basis_vectors += basis_vectors_to_add
        if not self._is_dummy_model:
            self._interpolations += interpolations_to_add
        # FALSE -> not masking, TRUE -> masking
        self._mask = np.append(self._mask, [False] * len(basis_vectors_to_add))

    def add_dof(  # pylint: disable=too-many-arguments
        self,
        cart_displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        interpolation_order: int,
        include_ref_polarizability: bool = True,
    ) -> None:
        """Add a degree of freedom (DOF).

        Specification of a DOF requires a displacement (directions the atoms move)
        alongside displacement amplitudes and corresponding known polarizabilities for
        each amplitude. Alongside the DOF specified, all DOFs related by the system's
        symmetry will be added as well. The interpolation order can be specified,
        though one must ensure that sufficient data is available.

        Parameters
        ----------
        cart_displacement
            (Å) 2D array with shape (N,3) where N is the number of atoms.

            Magnitude is arbitrary. Must be orthogonal to all previously added DOFs.
        amplitudes
            (Å) 1D array with shape (L,).

            Duplicate amplitudes, either those explicitly provided or those generated
            by structural symmetries, will raise :class:`.InvalidDOFException`.
        polarizabilities
            3D array with shape (1,3,3) or (2,3,3) containing known
            polarizabilities for each amplitude.

            If dummy model, this parameter is ignored.
        interpolation_order
            | Must be less than the number of total number of amplitudes after
            | symmetry considerations.
        include_ref_polarizability
            | Whether to include the references polarizability at 0.0 amplitude in the
            | interpolation.

        Raises
        ------
        InvalidDOFException
            Provided degree of freedom was invalid.

        """
        try:
            displacement = self.ref_structure.get_frac_displacement(cart_displacement)

            parent_displacement = displacement / np.linalg.norm(displacement * 10.0)
        except TypeError as exc:
            raise TypeError("displacement is not an ndarray") from exc
        verify_ndarray_shape("amplitudes", amplitudes, (None,))
        if self._is_dummy_model:
            polarizabilities = np.zeros((len(amplitudes), 3, 3))
        verify_ndarray_shape(
            "polarizabilities", polarizabilities, (len(amplitudes), 3, 3)
        )

        # Get information needed for DOF - checks displacement.
        basis_vectors_to_add, interpolation_xs, interpolation_ys = self._get_dof(
            parent_displacement,
            amplitudes,
            polarizabilities,
            include_ref_polarizability,
        )

        # Then append the DOF - checks amplitudes (in form of interpolation_xs)
        self._construct_and_add_interpolations(
            basis_vectors_to_add,
            interpolation_xs,
            interpolation_ys,
            interpolation_order,
        )

    def add_dof_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
        interpolation_order: int,
    ) -> None:
        """Add a degree of freedom (DOF) from file(s).

        Required displacements, amplitudes, and polarizabilities are automatically
        determined from provided files. Files should be chosen such that the
        resulting DOFs are valid under the same restrictions of :meth:`add_dof`.

        Parameters
        ----------
        filepaths
        file_format
            Supports ``"outcar"`` and ``"vasprun.xml"`` (see :ref:`Supported formats`).

            If dummy model, supports ``"poscar"`` and ``"xdatcar"`` as well.

        Raises
        ------
        FileNotFoundError
            File not found.
        InvalidFileException
            Invalid file.
        InvalidDOFException
            DOF assembled from supplied files was invalid. See :meth:`add_dof` for
            restrictions.


        """
        # Checks displacements
        displacements, amplitudes, polarizabilities = self._read_dof(
            filepaths, file_format
        )

        # Checks amplitudes
        self.add_dof(
            self.ref_structure.get_cart_displacement(displacements[0]),
            amplitudes,
            polarizabilities,
            interpolation_order,
        )

    def _read_dof(
        self, filepaths: str | Path | list[str] | list[Path], file_format: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Read displacements, amplitudes, and polarizabilities from file(s).

        This function does not change the state of the model.

        Parameters
        ----------
        filepaths:
            Supports: "outcar". If dummy model, supports: "outcar", "poscar" (see
            :ref:`Supported formats`).

        Returns
        -------
        :
            3-tuple with the form (displacements, polarizabilities, basis vector)

        Raises
        ------
        FileNotFoundError
            File could not be found.
        InvalidDOFException
            DOF assembled from supplied files was invalid (see get_dof)
        """
        displacements = []
        polarizabilities = []
        filepaths = pathify_as_list(filepaths)
        for filepath in filepaths:
            if not self._is_dummy_model:
                positions, polarizability = (
                    generic_io.read_positions_and_polarizability(filepath, file_format)
                )
            else:
                positions = generic_io.read_positions(filepath, file_format)
                polarizability = np.zeros((3, 3))

            try:
                displacement = calc_displacement(
                    self._ref_structure.positions,
                    positions,
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
        cart_basis_vector = self._ref_structure.get_cart_displacement(displacements[0])
        cart_basis_vector /= np.linalg.norm(cart_basis_vector)

        # Calculate amplitudes
        amplitudes = []
        for displacement in displacements:
            cart_displacement = self._ref_structure.get_cart_displacement(displacement)
            amplitudes.append(get_amplitude(cart_basis_vector, cart_displacement))
        return (
            np.array(displacements),
            np.array(amplitudes),
            np.array(polarizabilities),
        )

    def get_masked_model(self, dof_indexes_to_mask: list[int]) -> InterpolationModel:
        """Return new model with certain degrees of freedom deactivated.

        Model masking allows for the calculation of partial Raman spectra in which only
        certain degrees of freedom are considered.
        """
        result = copy.deepcopy(self)
        new_mask = result.mask
        new_mask[:] = False
        new_mask[dof_indexes_to_mask] = True
        result.mask = new_mask
        return result

    def unmask(self) -> None:
        """Clear mask, activating all specified DOFs."""
        self._mask[:] = False

    def __repr__(self) -> str:
        """Return string representation."""
        total_dofs = 3 * len(self._ref_structure.positions)
        specified_dofs = len(self._cart_basis_vectors)
        core = f"{specified_dofs}/{total_dofs}"
        if specified_dofs == total_dofs:
            core = ANSICOLORS.OK_GREEN + core + ANSICOLORS.END
        elif 1 <= specified_dofs < total_dofs:
            core = ANSICOLORS.WARNING_YELLOW + core + ANSICOLORS.END
        else:
            core = ANSICOLORS.ERROR_RED + core + ANSICOLORS.END

        result = f"InterpolationModel with {core} degrees of freedom specified."

        num_masked = np.sum(self._mask)
        if num_masked > 0:
            num_arts = len(self._cart_basis_vectors)
            msg = f"ATTENTION: {num_masked}/{num_arts} degrees of freedom are masked."
            result += f"\n {ANSICOLORS.WARNING_YELLOW} {msg} {ANSICOLORS.END}"
        if self._is_dummy_model:
            msg = "ATTENTION: this is a dummy model."
            result += f"\n {ANSICOLORS.WARNING_YELLOW} {msg} {ANSICOLORS.END}"

        return result
