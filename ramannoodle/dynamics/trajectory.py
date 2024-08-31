"""Molecular dynamics trajectories."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ramannoodle.dynamics.abstract import Dynamics
from ramannoodle.polarizability.abstract import PolarizabilityModel
from ramannoodle.exceptions import verify_ndarray_shape, get_type_error
from ramannoodle.spectrum.raman import MDRamanSpectrum
from ramannoodle.structure.structure_utils import apply_pbc


class Trajectory(Dynamics, Sequence[NDArray[np.float64]]):
    r"""Molecular dynamics trajectory.

    A trajectory is represented as a positions time series (``positions_ts``) and a
    timestep.

    Parameters
    ----------
    positions_ts
        | (fractional) 3D array with shape (S,N,3) where S in the number of
        | configurations and N is the number of atoms.
    timestep
        | (fs)

    """

    def __init__(
        self,
        positions_ts: NDArray[np.float64],
        timestep: float,
    ) -> None:
        verify_ndarray_shape("positions_ts", positions_ts, (None, None, 3))
        try:
            timestep = float(timestep)
        except TypeError as exc:
            raise get_type_error("timestep", timestep, "float") from exc
        if timestep <= 0:
            raise ValueError("timestep must be positive")

        self._positions_ts = apply_pbc(positions_ts)
        self._timestep = timestep

    @property
    def positions_ts(self) -> NDArray[np.float64]:
        r"""Get (a copy of) the positions time series.

        Returns
        -------
        :
            (fractional) 3D array with shape (S,N,3) where S in the number of
            configurations and N is the number of atoms.
        """
        return self._positions_ts.copy()

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
            | Must be compatible with the trajectory.
        """
        try:
            polarizability_ts = polarizability_model.calc_polarizabilities(
                self._positions_ts
            )
        except ValueError as exc:
            raise ValueError(
                "polarizability_model and trajectory are incompatible"
            ) from exc

        return MDRamanSpectrum(polarizability_ts, self._timestep)

    def __len__(self) -> int:
        """Get trajectory length in timesteps."""
        return len(self._positions_ts)

    @overload
    def __getitem__(self, key: int) -> NDArray[np.float64]: ...

    @overload
    def __getitem__(self, key: slice) -> NDArray[np.float64]: ...

    def __getitem__(self, key: int | slice) -> NDArray[np.float64]:
        """Get positions (supports indexing and slicing)."""
        try:
            return self._positions_ts[key]
        except IndexError as exc:
            if "out of bounds" in str(exc):
                raise IndexError("trajectory index out of bounds") from exc
            raise exc
