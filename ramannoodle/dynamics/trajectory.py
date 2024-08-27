"""Molecular dynamics trajectories."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.abstract import Dynamics
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.exceptions import verify_ndarray_shape
from ramannoodle.spectrum.raman import MDRamanSpectrum


class Trajectory(Dynamics):
    r"""Trajectory from molecular dynamics.

    Parameters
    ----------
    cart_displacement_ts
        Unitless | 3D array with shape (S,N,3) where S in the number of configurations
        and N is the number of atoms.
    timestep
        In fs.

    """

    def __init__(
        self,
        cart_displacement_ts: NDArray[np.float64],
        timestep: float,
    ) -> None:
        verify_ndarray_shape(
            "cart_displacements_ts", cart_displacement_ts, (None, None, 3)
        )
        self._cart_displacement_ts = cart_displacement_ts
        self._timestep = timestep

    @property
    def cart_displacement_ts(self) -> NDArray[np.float64]:
        r"""Get (a copy of) the cartesian displacement time series.

        Returns
        -------
        :
            Unitless | 3D array with shape (S,N,3) where S in the number of
            molecular dynamics snapshots and N is the number of atoms.
        """
        return self._cart_displacement_ts.copy()

    @property
    def timestep(self) -> float:
        """Get timestep in fs."""
        return self._timestep

    def get_raman_spectrum(
        self, polarizability_model: PolarizabilityModel
    ) -> MDRamanSpectrum:
        """Calculate a raman spectrum using a polarizability model.

        Parameters
        ----------
        polarizability_model
            Must be compatible with the trajectory.
        """
        try:
            polarizability_ts = polarizability_model.get_polarizability(
                self._cart_displacement_ts
            )
        except ValueError as exc:
            raise ValueError(
                "polarizability_model and trajectory are incompatible"
            ) from exc

        return MDRamanSpectrum(polarizability_ts, self._timestep)
