"""Harmonic lattice vibrations aka phonons."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.abstract import Dynamics
from ramannoodle.globals import RAMAN_TENSOR_CENTRAL_DIFFERENCE
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.spectrum.raman import PhononRamanSpectrum
from ramannoodle.exceptions import verify_ndarray_shape


class Phonons(Dynamics):
    r"""Harmonic lattice vibrations.

    A phonon can be represented by a wavenumber and corresponding atomic displacement.
    The wavenumbers are the eigenvalues of the system's dynamical matrix, while the
    atomic displacements are the eigenvectors of the dynamical matrix divided by the
    square root of the atomic masses.

    Parameters
    ----------
    ref_positions
        | (fractional) 2D array with shape (N,3) where N is the number of atoms.
    wavenumbers
        | (cm\ :sup:`-1`) 1D array with shape (M,).
    displacements
        | (fractional) 3D array with shape (M,N,3).
    """

    def __init__(
        self,
        ref_positions: NDArray[np.float64],
        wavenumbers: NDArray[np.float64],
        displacements: NDArray[np.float64],
    ) -> None:
        verify_ndarray_shape("ref_positions", ref_positions, (None, 3))
        verify_ndarray_shape("wavenumbers", wavenumbers, (None,))
        verify_ndarray_shape(
            "displacements",
            displacements,
            (wavenumbers.size, ref_positions.shape[0], 3),
        )
        self._ref_positions = ref_positions
        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._displacements: NDArray[np.float64] = displacements

    @property
    def ref_positions(self) -> NDArray[np.float64]:
        r"""Get (a copy of) reference positions.

        Returns
        -------
        :
            (fractional) 2D array with shape (N,3) where N is the number of atoms.
        """
        return self._ref_positions.copy()

    @property
    def wavenumbers(self) -> NDArray[np.float64]:
        r"""Get (a copy of) wavenumbers.

        Returns
        -------
        :
            (cm\ :sup:`-1`) 1D array with shape (M,) where M is the number of phonons.
        """
        return self._wavenumbers.copy()

    @property
    def displacements(self) -> NDArray[np.float64]:
        """Get (a copy of) displacements.

        Returns
        -------
        :
            (fractional) 3D array with shape (M,N,3) where M is the number of phonons
            and N is the number of atoms.
        """
        return self._displacements.copy()

    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a raman spectrum using a polarizability model.

        Parameters
        ----------
        polarizability_model
            | Must be compatible with phonons.
        """
        raman_tensors = []
        for displacement in self._displacements:
            try:
                epsilon = displacement * RAMAN_TENSOR_CENTRAL_DIFFERENCE
                plus = polarizability_model.calc_polarizabilities(
                    np.array([self.ref_positions + epsilon])
                )[0]
                minus = polarizability_model.calc_polarizabilities(
                    np.array([self.ref_positions - epsilon])
                )[0]
            except ValueError as exc:
                raise ValueError(
                    "polarizability_model and phonons are incompatible"
                ) from exc
            raman_tensors.append((plus - minus) / RAMAN_TENSOR_CENTRAL_DIFFERENCE)

        return PhononRamanSpectrum(self._wavenumbers, np.array(raman_tensors))
