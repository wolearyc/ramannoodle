"""Class for phonons."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.abstract import Dynamics
from ramannoodle.globals import RAMAN_TENSOR_CENTRAL_DIFFERENCE
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.spectrum.raman import PhononRamanSpectrum


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
        try:
            if wavenumbers.ndim != 1:
                raise ValueError("wavenumbers is not a 1D array")
        except AttributeError as exc:
            raise TypeError("wavenumbers is not an ndarray") from exc
        try:
            if cart_displacements.ndim != 3 or cart_displacements.shape[2] != 3:
                raise ValueError("cart_displacements does not have shape (_,_,3)")
            if cart_displacements.shape[0] != wavenumbers.shape[0]:
                raise ValueError(
                    "wavenumbers and cart_displacements do not have the same length"
                )
        except AttributeError as exc:
            raise TypeError("cart_displacements is not an ndarray") from exc

        self._wavenumbers: NDArray[np.float64] = wavenumbers
        self._cart_displacements: NDArray[np.float64] = cart_displacements

    @property
    def wavenumbers(self) -> NDArray[np.float64]:
        """Get wavenumbers.

        Returns
        -------
        :
            1D array with length M
        """
        return self._wavenumbers

    @property
    def cart_displacements(self) -> NDArray[np.float64]:
        """Get cartesian displacements.

        Returns
        -------
        :
            3D array with shape (M,N,3) where M is the number of displacements
            and N is the number of atoms
        """
        return self._cart_displacements

    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> PhononRamanSpectrum:
        """Calculate a raman spectrum using a polarizability model."""
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
                    "polarizability_model is incompatible with phonons"
                ) from exc
            raman_tensors.append((plus - minus) / RAMAN_TENSOR_CENTRAL_DIFFERENCE)

        return PhononRamanSpectrum(self._wavenumbers, np.array(raman_tensors))
