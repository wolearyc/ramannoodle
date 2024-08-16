"""Phonons aka. harmonic lattice vibrations."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.abstract import Dynamics
from ramannoodle.globals import RAMAN_TENSOR_CENTRAL_DIFFERENCE
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.spectrum.raman import PhononRamanSpectrum
from ramannoodle.exceptions import verify_ndarray_shape


class Phonons(Dynamics):
    """Harmonic lattice vibrations.

    A phonon can be represented by a wavenumber and corresponding atomic displacement.
    The wavenumbers are the eigenvalues of the system's dynamical matrix, while the
    atomic displacements are the eigenvectors of the dynamical matrix divided by the
    square root of the atomic masses.

    Parameters
    ----------
    wavenumbers
        1D array with length M
    cart_displacements
        3D array with shape (M,N,3) where N is the number of atoms

    """

    def __init__(
        self,
        wavenumbers: NDArray[np.float64],
        cart_displacements: NDArray[np.float64],
    ) -> None:
        verify_ndarray_shape("wavenumbers", wavenumbers, (None,))
        verify_ndarray_shape(
            "cart_displacements", cart_displacements, (wavenumbers.size, None, 3)
        )
        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._cart_displacements: NDArray[np.float64] = cart_displacements

    @property
    def wavenumbers(self) -> NDArray[np.float64]:
        """Get (a copy of) wavenumbers.

        Returns
        -------
        :
            1D array with shape (M,) where M is the number of phonons.
        """
        return self._wavenumbers.copy()

    @property
    def cart_displacements(self) -> NDArray[np.float64]:
        """Get (a copy of) cartesian displacements.

        Returns
        -------
        :
            3D array with shape (M,N,3) where M is the number of phonons
            and N is the number of atoms
        """
        return self._cart_displacements.copy()

    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a raman spectrum using a polarizability model.

        Parameters
        ----------
        polarizability_model
            must be compatible with phonons
        """
        raman_tensors = []
        for cart_displacement in self._cart_displacements:
            try:
                plus = polarizability_model.get_polarizability(
                    cart_displacement * RAMAN_TENSOR_CENTRAL_DIFFERENCE
                )
                minus = polarizability_model.get_polarizability(
                    -cart_displacement * RAMAN_TENSOR_CENTRAL_DIFFERENCE
                )
            except ValueError as exc:
                raise ValueError(
                    "polarizability_model and phonons are incompatible"
                ) from exc
            raman_tensors.append((plus - minus) / RAMAN_TENSOR_CENTRAL_DIFFERENCE)

        return PhononRamanSpectrum(self._wavenumbers, np.array(raman_tensors))
