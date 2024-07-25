"""Parent class for polarizability models."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline, BSpline

from . import polarizability_utils
from ..exceptions import InvalidDOFException


class PolarizabilityModel(ABC):  # pylint: disable=too-few-public-methods
    """Represents a polarizability model"""

    @abstractmethod
    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Returns a polarizability for a set of atomic displacements."""


class InterpolationPolarizabilityModel(PolarizabilityModel):
    """This model uses interpolation around independent degrees of
    freedom to estimate polarizabilities.
    With linear interpolation + phonon displacements, we get a standard
    raman-tensor-based spectrum.
    With linear interpolation + site displacements, we get an atomic raman
    tensor-based spectrum. This class also needs to take careful care
    to obey symmetry operations.

    """

    def __init__(self) -> None:  # - Generate all symmetry equivalent displacements.
        self._dof_displacements: list[NDArray[np.float64]] = []
        self._dof_interpolations: list[BSpline] = []

    def get_polarizability(
        self, displacement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Returns a polarizability for a given displacement."""

        # Project displacement onto each dof_displacement, use the coordinate to
        # define the interpolation.

        return np.array([])

    def add_dof(
        self,
        displacements_and_polarizabilities: list[
            tuple[NDArray[np.float64], NDArray[np.float64]]
        ],
        interpolation_dim: int,
    ) -> None:
        """Adds a degree of freedom."""
        assert len(displacements_and_polarizabilities) >= 2
        assert interpolation_dim < len(displacements_and_polarizabilities)

        dof_displacement, _ = displacements_and_polarizabilities[0]
        dof_displacement /= np.linalg.norm(dof_displacement)

        # Check that all displacements are collinear
        for i, (displacement, _) in enumerate(displacements_and_polarizabilities):
            if not polarizability_utils.are_collinear(dof_displacement, displacement):
                raise InvalidDOFException(f"displacement (index={i}) is not collinear")

        # Check that new displacement is orthogonal with existing displacements
        result = polarizability_utils.check_orthogonal(
            dof_displacement, self._dof_displacements
        )
        if result != -1:
            raise InvalidDOFException(
                f"new dof is not orthogonal with existing dof (index={result})"
            )

        interpolation = make_interp_spline(
            x=[d for d, _ in displacements_and_polarizabilities],
            y=[p for _, p in displacements_and_polarizabilities],
            k=interpolation_dim,
            bc_type="natural",
        )

        self._dof_displacements.append(dof_displacement)
        self._dof_interpolations.append(interpolation)
