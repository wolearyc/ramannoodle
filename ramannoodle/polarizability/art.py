"""Polarizability model based on atomic Raman tensors (ARTs)."""

from __future__ import annotations
from typing import cast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from ramannoodle.globals import ANSICOLORS
from ramannoodle.polarizability.interpolation import InterpolationModel
from ramannoodle.exceptions import (
    get_type_error,
    get_shape_error,
    verify_ndarray_shape,
    InvalidDOFException,
    UsageError,
)


def _get_directions_str(directions: list[NDArray[np.float64]]) -> str:
    """Return easy-to-read string for directions."""
    direction_strs = []
    for direction in directions:
        direction_strs.append(
            np.array2string(direction, precision=5, suppress_small=True, sign="+")
        )
    return ", ".join(direction_strs)


def _get_specified_str(num_specified: int) -> str:
    """Return ansi-colored string for number of specified models."""
    core = f"{num_specified}/3"
    if num_specified == 3:
        return ANSICOLORS.OK_GREEN + core + ANSICOLORS.END
    if 1 <= num_specified <= 2:
        return ANSICOLORS.WARNING_YELLOW + core + ANSICOLORS.END
    return ANSICOLORS.ERROR_RED + core + ANSICOLORS.END


class ARTModel(InterpolationModel):
    """Polarizability model based on atomic Raman tensors (ARTs).

    This model is a flavor of :class:`.InterpolationModel` with added restrictions. The
    degrees of freedom (DOF) in ARTModel are displacements of individual atoms. 1D
    interpolations are used, with only two amplitudes allowed per DOF. Atomic
    displacements can be in any direction, so long as all three directions for a given
    atom are orthogonal.

    All assumptions from :class:`.InterpolationModel` apply.

    .. note::
        Degrees of freedom cannot (and should not) be added using the :meth:`.add_dof`
        and :meth:`.add_dof_from_files` methods inherited from
        :class:`.InterpolationModel`. Usage of these methods will raise a
        :class:`.UsageError`. Instead, use :meth:`add_art` or
        :meth:`add_art_from_files`.

    .. warning::
        Due to the way ARTModel is implemented, the predicted tensor of the reference
        structure may be offset slightly. This is a fixed offset and therefore has no
        influence on Raman spectra calculations.

    Parameters
    ----------
    ref_structure
        | Reference structure on which to base the model.
    ref_polarizability
        | 2D array with shape (3,3) giving polarizability of the reference structure.
    is_dummy_model

    """

    def add_dof(  # pylint: disable=too-many-arguments
        self,
        cart_displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        interpolation_order: int,
        include_ref_polarizability: bool = True,
    ) -> None:
        """Disable add_dof.

        Raises
        ------
        UsageError

        :meta private:
        """
        raise UsageError("add_dof should not be used; use add_art instead")

    def add_dof_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
        interpolation_order: int,
    ) -> None:
        """Disable add_dof_from_files.

        Raises
        ------
        UsageError

        :meta private:
        """
        raise UsageError(
            "add_dof_from_files should not be used; use add_art_from_files instead"
        )

    def add_art(
        self,
        atom_index: int,
        cart_direction: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
    ) -> None:
        """Add an atomic Raman tensor (ART).

        Specification of an ART requires an atom index, displacement direction,
        displacement amplitudes, and corresponding known polarizabilities for each
        amplitude. ARTs related to the specified ART by symmetry are added
        automatically.

        Parameters
        ----------
        atom_index
            | Atom index in the reference structure.
        cart_direction
            (Å) 1D array with shape (3,).

            Must be orthogonal to all previously added ARTs belonging to the atom at
            ``atom_index``.
        amplitudes
            (Å) 1D array with shape (1,) or (2,).

            Duplicate amplitudes, either those explicitly provided or those generated
            by structural symmetries, will raise :class:`.InvalidDOFException`.
        polarizabilities
            3D array with shape (1,3,3) or (2,3,3) containing known
            polarizabilities for each amplitude.

            If dummy model, this parameter is ignored.

        Raises
        ------
        InvalidDOFException
            Provided ART was invalid.

        """
        direction = self.ref_structure.get_frac_direction(cart_direction)
        if not isinstance(atom_index, int):
            raise get_type_error("atom_index", atom_index, "int")
        try:
            if amplitudes.shape not in [(1,), (2,)]:
                raise get_shape_error("amplitudes", amplitudes, "(1,) or (2,)")
        except AttributeError as exc:
            raise get_type_error("amplitudes", amplitudes, "ndarray") from exc
        if self._is_dummy_model:
            polarizabilities = np.zeros((amplitudes.size, 3, 3))
        verify_ndarray_shape(
            "polarizabilities", polarizabilities, (amplitudes.size, 3, 3)
        )

        displacement = self._ref_structure.positions * 0
        displacement[atom_index] = direction / np.linalg.norm(direction * 10.0)

        # Checks amplitudes and displacements
        super().add_dof(
            self.ref_structure.get_cart_displacement(displacement),
            amplitudes,
            polarizabilities,
            1,
            include_ref_polarizability=False,
        )

    def add_art_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
    ) -> None:
        """Add an atomic Raman tensor (ART) from file(s).

        Required directions, amplitudes, and polarizabilities are automatically
        determined from provided files. Files should be chosen such that the
        resulting ARTs are valid under the same restrictions of :meth:`add_art`.

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
            ART assembled from supplied files was invalid. See :meth:`add_art` for
            restrictions.

        """
        displacements, amplitudes, polarizabilities = super()._read_dof(
            filepaths, file_format
        )
        # Check whether only one atom is displaced.
        _displacement = displacements[0].copy()
        atom_index = int(np.argmax(np.sum(_displacement**2, axis=1)))
        _displacement[atom_index] = np.zeros(3)
        if not np.isclose(_displacement, 0.0, atol=1e-6).all():
            raise InvalidDOFException("multiple atoms displaced simultaneously")

        # Checks displacement
        basis_vectors_to_add, interpolation_xs, interpolation_ys = super()._get_dof(
            displacements[0], amplitudes, polarizabilities, False
        )

        num_amplitudes = len(interpolation_xs[0])
        if num_amplitudes != 2:
            raise InvalidDOFException(
                f"wrong number of amplitudes: {num_amplitudes} != 2"
            )

        # Checks amplitudes
        super()._construct_and_add_interpolations(
            basis_vectors_to_add, interpolation_xs, interpolation_ys, 1
        )

    def get_specification_tuples(
        self,
    ) -> list[tuple[int, list[int], list[NDArray[np.float64]]]]:
        """Return tuples with information on model.

        Returns
        -------
        :
            List of 3-tuples:
                0. An atom index, referred to below as ``parent_atom_index``
                1. List of atom indexes that are symmetrically equivalent to
                   ``parent_atom_index``
                #. | Currently specified ART directions for ``parent_atom_index``
                   | (Å) List of 1D arrays with shape (3,).

        """
        equivalent_atom_dict = self._ref_structure.get_equivalent_atom_dict()

        specification_tuples = []
        for atom_index in equivalent_atom_dict:
            specification_tuples.append(
                (
                    atom_index,
                    equivalent_atom_dict[atom_index],
                    self._get_art_cart_directions(atom_index),
                )
            )
        return specification_tuples

    def get_dof_indexes(
        self, atom_indexes_or_symbols: int | str | list[int | str]
    ) -> list[int]:
        """Return the DOF indexes of certain atoms.

        Parameters
        ----------
        atom_indexes_or_symbols
            If integer or list of integers, specifies atom indexes. If string or list
            of strings, specifies atom symbols.

            Mixtures of indexes and symbols are allowed.

        """
        if not isinstance(atom_indexes_or_symbols, list):
            atom_indexes_or_symbols = list([atom_indexes_or_symbols])

        atom_indexes = []
        for item in atom_indexes_or_symbols:
            if isinstance(item, str):
                atom_indexes += self._ref_structure.get_atom_indexes(item)
            else:
                atom_indexes += [item]
        atom_indexes = list(set(atom_indexes))

        dof_indexes = []
        for atom_index in atom_indexes:
            for index, basis_vector in enumerate(self._cart_basis_vectors):
                direction = basis_vector[atom_index]
                if not np.isclose(direction, 0, atol=1e-5).all():
                    dof_indexes.append(index)
        return dof_indexes

    def _get_art_cart_directions(self, atom_index: int) -> list[NDArray[np.float64]]:
        """Return specified art cartesian direction vectors for an atom."""
        indexes = self.get_dof_indexes(atom_index)
        cart_directions = [
            self._cart_basis_vectors[index][atom_index] for index in indexes
        ]
        return cart_directions

    def __repr__(self) -> str:
        """Get string representation."""
        specification_tuples = self.get_specification_tuples()

        table = [["Atom index", "Directions", "Specified", "Equivalent atoms"]]
        for (
            atom_index,
            equivalent_atom_indexes,
            cart_directions,
        ) in specification_tuples:
            row = [str(atom_index)]
            row.append(_get_directions_str(cart_directions))
            row.append(_get_specified_str(len(cart_directions)))
            row.append(str(len(equivalent_atom_indexes)))
            table.append(row)
        result = tabulate(table, headers="firstrow", tablefmt="rounded_outline")

        num_masked = np.sum(self._mask)
        if num_masked > 0:
            num_arts = len(self._cart_basis_vectors)
            msg = f"ATTENTION: {num_masked}/{num_arts} atomic Raman tensors are masked."
            result += f"\n {ANSICOLORS.WARNING_YELLOW} {msg} {ANSICOLORS.END}"
        if self._is_dummy_model > 0:
            msg = "ATTENTION: this is a dummy model."
            result += f"\n {ANSICOLORS.WARNING_YELLOW} {msg} {ANSICOLORS.END}"

        return result

    def get_masked_model(self, dof_indexes_to_mask: list[int]) -> ARTModel:
        """Return new model with certain degrees of freedom masked (deactivated).

        Model masking allows for the calculation of partial Raman spectra in which only
        certain degrees of freedom are considered.

        Parameters
        ----------
        dof_indexes_to_mask
            | DOF indexes associated with specific atoms can be retrieved using
            | :meth:`get_dof_indexes`.
        """
        # We "cast" here due to how typing is done to support Python 3.10.
        return cast(ARTModel, super().get_masked_model(dof_indexes_to_mask))
