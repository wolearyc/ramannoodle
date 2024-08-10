"""Polarizability model based on atomic Raman tensors (ARTs)."""

from __future__ import annotations
from typing import cast

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

from ..globals import AnsiColors
from .interpolation import InterpolationModel
from ..exceptions import (
    get_type_error,
    get_shape_error,
    verify_ndarray_shape,
    InvalidDOFException,
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
        return AnsiColors.OK_GREEN + core + AnsiColors.END
    if 1 <= num_specified <= 2:
        return AnsiColors.WARNING_YELLOW + core + AnsiColors.END
    return AnsiColors.ERROR_RED + core + AnsiColors.END


class ARTModel(InterpolationModel):
    """Polarizability model based on atomic Raman tensors (ARTs).

    This model is a flavor of InterpolationModel with added restrictions. The degrees
    of freedom (DOF) in ARTModel are displacements of individual atoms. 1D
    interpolations are used, with only two amplitudes allowed per DOF. Atomic
    displacements can be in any direction, so long as all three directions for a given
    atom are orthogonal.

    All assumptions from InterpolationModel apply.

    .. note::
        Degrees of freedom cannot (and should not) be added using `add_dof` and
        `add_dof_from_files`. Instead, use `add_art` and `add_art_from_files`.

    .. warning::
        Due to the way InterpolationModel works, an ARTModel's predicted polarizability
        tensor of the equilibrium structure may be offset slightly. This is a fixed
        offset and therefore has no influence on Raman spectra calculations.

    Parameters
    ----------
    structural_symmetry
    equilibrium_polarizability
        2D array with shape (3,3) giving polarizability of system at equilibrium. This
        would usually correspond to the minimum energy structure.

    """

    def add_dof(  # pylint: disable=too-many-arguments
        self,
        displacement: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        interpolation_order: int,
        include_equilibrium_polarizability: bool = True,
    ) -> None:
        """Disable add_dof.

        :meta private:
        """
        raise AttributeError("'ARTModel' object has no attribute 'add_dof'")

    def add_dof_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
        interpolation_order: int,
    ) -> None:
        """Disable add_dof_from_files.

        :meta private:
        """
        raise AttributeError("'ARTModel' object has no attribute 'add_dof_from_files'")

    def add_art(
        self,
        atom_index: int,
        direction: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
    ) -> None:
        """Add an atomic Raman tensor (ART).

        Specification of an ART requires an atom index, displacement direction,
        displacement amplitudes, and corresponding known polarizabilities for each
        amplitude. Alongside ART's related to the specified ART by symmetry are added
        automatically.

        Parameters
        ----------
        atom_index
            Index of atom, consistent with the StructuralSymmetry used to initialize
            the model.
        direction
            1D array with shape (3,). Must be orthogonal to all previously added ARTs
            belonging to specified atom.
        amplitudes
            1D array of length 1 or 2 containing amplitudes in angstroms. Duplicate
            amplitudes are not allowed, including symmetrically equivalent
            amplitudes.
        polarizabilities
            3D array with shape (1,3,3) or (2,3,3) containing known polarizabilities for
            each amplitude.

        Raises
        ------
        InvalidDOFException
            Provided ART was invalid.

        """
        if not isinstance(atom_index, int):
            raise get_type_error("atom_index", atom_index, "int")
        try:
            if amplitudes.shape not in [(1,), (2,)]:
                raise get_shape_error("amplitudes", amplitudes, "(1,) or (2,)")
        except AttributeError as exc:
            raise get_type_error("amplitudes", amplitudes, "ndarray") from exc
        verify_ndarray_shape(
            "polarizabilities", polarizabilities, (amplitudes.size, 3, 3)
        )

        displacement = self._structural_symmetry.get_fractional_positions() * 0
        try:
            displacement[atom_index] = direction / np.linalg.norm(direction * 10.0)
        except TypeError as exc:
            raise get_type_error("direction", direction, "ndarray") from exc
        except ValueError as exc:
            raise get_shape_error("direction", direction, "(3,)") from exc

        super().add_dof(
            displacement,
            amplitudes,
            polarizabilities,
            1,
            include_equilibrium_polarizability=False,
        )

    def add_art_from_files(
        self,
        filepaths: str | Path | list[str] | list[Path],
        file_format: str,
    ) -> None:
        """Add a atomic Raman tensor (ART) from file(s).

        Required directions, amplitudes, and polarizabilities are automatically
        determined from provided files. Files should be chosen wisely such that the
        resulting ARTs are valid under the restrictions set by `add_art`.

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
            ART assembled from supplied files was invalid (see get_art)

        """
        displacements, amplitudes, polarizabilities = super()._read_dof(
            filepaths, file_format
        )
        # Check whether only one atom is displaced.
        _displacement = np.copy(displacements[0])
        atom_index = int(np.argmax(np.sum(_displacement**2, axis=1)))
        _displacement[atom_index] = np.zeros(3)
        if not np.isclose(_displacement, 0.0, atol=1e-6).all():
            raise InvalidDOFException("multiple atoms displaced simultaneously")

        basis_vectors_to_add, interpolation_xs, interpolation_ys = super()._get_dof(
            displacements[0], amplitudes, polarizabilities, False
        )

        num_amplitudes = len(interpolation_xs[0])
        if num_amplitudes != 2:
            raise InvalidDOFException(
                f"wrong number of amplitudes: {num_amplitudes} != 2"
            )

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
            3-tuple. First element is an atom index, second element is a list of atom
            indexes that are symmetrically equivalent, and the third element is a list
            currently specified ART directions.

        """
        equivalent_atom_dict = self._structural_symmetry.get_equivalent_atom_dict()

        specification_tuples = []
        for atom_index in equivalent_atom_dict:
            specification_tuples.append(
                (
                    atom_index,
                    equivalent_atom_dict[atom_index],
                    self._get_art_directions(atom_index),
                )
            )
        return specification_tuples

    def get_dof_indexes(
        self, atom_indexes_or_symbols: int | str | list[int | str]
    ) -> list[int]:
        """Return art (DOF) indexes for certain atoms.

        Parameters
        ----------
        atom_indexes_or_symbols
            If integer or list of integers, specifies atom indexes. If string or list
            of strings, specifies atom symbols. Mixtures of integers and strings are
            allowed.

        """
        if not isinstance(atom_indexes_or_symbols, list):
            atom_indexes_or_symbols = list([atom_indexes_or_symbols])

        atom_indexes = []
        for item in atom_indexes_or_symbols:
            if isinstance(item, str):
                atom_indexes += self._structural_symmetry.get_atom_indexes(item)
            else:
                atom_indexes += [item]
        atom_indexes = list(set(atom_indexes))

        dof_indexes = []
        for atom_index in atom_indexes:
            for index, basis_vector in enumerate(self._cartesian_basis_vectors):
                direction = basis_vector[atom_index]
                if not np.isclose(direction, 0, atol=1e-5).all():
                    dof_indexes.append(index)
        return dof_indexes

    def _get_art_directions(self, atom_index: int) -> list[NDArray[np.float64]]:
        """Return specified art direction vectors for an atom."""
        indexes = self.get_dof_indexes(atom_index)
        directions = [
            self._cartesian_basis_vectors[index][atom_index] for index in indexes
        ]
        return directions

    def __repr__(self) -> str:
        """Get string representation."""
        specification_tuples = self.get_specification_tuples()

        table = [["Atom index", "Directions", "Specified", "Equivalent atoms"]]

        for atom_index, equivalent_atom_indexes, directions in specification_tuples:
            row = [str(atom_index)]
            row.append(_get_directions_str(directions))
            row.append(_get_specified_str(len(directions)))
            row.append(str(len(equivalent_atom_indexes)))
            table.append(row)

        result = tabulate(table, headers="firstrow", tablefmt="rounded_outline")
        return result

    def get_masked_model(self, dof_indexes_to_mask: list[int]) -> ARTModel:
        """Return new model with certain degrees of freedom deactivated.

        Model masking allows for the calculation of partial Raman spectra in which only
        certain degrees of freedom are considered.
        """
        # We "cast" here, due to how typing is done.
        return cast(ARTModel, super().get_masked_model(dof_indexes_to_mask))
