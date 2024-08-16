"""Raman spectrum classes and utilities."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import get_type_error, verify_ndarray_shape
from ramannoodle.globals import BOLTZMANN_CONSTANT


def get_bose_einstein_correction(
    wavenumbers: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    r"""Calculate Bose-Einstein spectral correction.

    Parameters
    ----------
    wavenumbers
        cm\ :sup:`-1` | 1D array with shape (M,).
    temperature
        In Kelvin.

    Returns
    -------
    :
        Correction factor for each wavenumber.

    Raises
    ------
    ValueError

    """
    try:
        if temperature <= 0:
            raise ValueError(f"invalid temperature: {temperature} <= 0")
    except TypeError as exc:
        raise get_type_error("temperature", temperature, "float") from exc
    try:
        energy = wavenumbers * 29979245800.0 * 4.1357e-15  # in eV
        return 1 / (1 - np.exp(-energy / (BOLTZMANN_CONSTANT * temperature)))
    except TypeError as exc:
        raise get_type_error("wavenumbers", wavenumbers, "ndarray") from exc


def get_laser_correction(
    wavenumbers: NDArray[np.float64], laser_wavenumber: float
) -> NDArray[np.float64]:
    r"""Calculate conventional laser-wavenumber-dependent spectral correction.

    Parameters
    ----------
    wavenumbers
        cm\ :sup:`-1` | 1D array with shape (M,).
    laser_wavenumber
        In cm\ :sup:`-1`.

    Returns
    -------
    :
        Correction factor for each wavenumber.

    """
    try:
        if laser_wavenumber <= 0:
            raise ValueError(f"invalid laser_wavenumber: {laser_wavenumber} <= 0")
    except TypeError as exc:
        raise get_type_error("laser_wavenumber", laser_wavenumber, "float") from exc
    try:
        return ((wavenumbers - laser_wavenumber) / 10000) ** 4 / wavenumbers
    except TypeError as exc:
        raise get_type_error("wavenumbers", wavenumbers, "ndarray") from exc


class PhononRamanSpectrum:  # pylint: disable=too-few-public-methods
    r"""Phonon-based first-order Raman spectrum.

    Parameters
    ----------
    phonon_wavenumbers
        cm\ :sup:`-1` | 1D array with shape (M,) where M is the number of phonons.
    raman_tensors
        Unitless | 3D array with shape (M,3,3).

    """

    def __init__(
        self,
        phonon_wavenumbers: NDArray[np.float64],
        raman_tensors: NDArray[np.float64],
    ) -> None:
        verify_ndarray_shape("phonon_wavenumbers", phonon_wavenumbers, (None,))
        verify_ndarray_shape(
            "raman_tensors", raman_tensors, (len(phonon_wavenumbers), 3, 3)
        )
        self._phonon_wavenumbers = phonon_wavenumbers
        self._raman_tensors = raman_tensors

    @property
    def phonon_wavenumbers(self) -> NDArray[np.float64]:
        r"""Get (a copy of) phonon_wavenumbers.

        Returns
        -------
        :
            cm\ :sup:`-1` | 1D array with shape (M,) where M is the number of phonons.
        """
        return self._phonon_wavenumbers.copy()

    @property
    def raman_tensors(self) -> NDArray[np.float64]:
        """Get (a copy of) raman_tensors.

        Returns
        -------
        :
            Unitless | 3D array with shape (M,3,3) where M is the number of phonons.
        """
        return self._raman_tensors.copy()

    def measure(  # pylint: disable=too-many-arguments
        self,
        orientation: str | NDArray[np.float64] = "polycrystalline",
        laser_correction: bool = False,
        laser_wavelength: float = 522,
        bose_einstein_correction: bool = False,
        temperature: float = 300,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Calculate and return a raw Raman spectrum.

        Parameters
        ----------
        orientation
            Currently only "polycrystalline" is supported. Future versions will support
            arbitrary orientations.
        laser_correction
            Applies laser-wavelength-dependent intensity correction. If True,
            ``laser_wavelength`` must be specified.
        laser_wavelength
            In nm.
        bose_einstein_correction
            Applies temperature-dependent Bose Einstein correction. If True,
            ``temperature`` must be specified.
        temperature
            In Kelvin.

        Returns
        -------
        :
            2-tuple. First element is wavenumbers (cm\ :sup:`-1`), a 1D array with
            shape (M,) where M is the number of phonons. The second element is
            intensities (arbitrary units), a 1D array with shape (M,).

        Raises
        ------
        NotImplementedError
            Raised when any orientation besides "polycrystalline" is supplied.

        """
        if orientation != "polycrystalline":
            raise NotImplementedError(
                "only polycrystalline spectra are supported for now"
            )

        alpha_squared = (
            (
                self._raman_tensors[:, 0, 0]
                + self._raman_tensors[:, 1, 1]
                + self._raman_tensors[:, 2, 2]
            )
            / 3.0
        ) ** 2
        gamma_squared = (
            (self._raman_tensors[:, 0, 0] - self._raman_tensors[:, 1, 1]) ** 2
            + (self._raman_tensors[:, 0, 0] - self._raman_tensors[:, 2, 2]) ** 2
            + (self._raman_tensors[:, 1, 1] - self._raman_tensors[:, 2, 2]) ** 2
            + 6.0
            * (
                self._raman_tensors[:, 0, 1] ** 2
                + self._raman_tensors[:, 0, 2] ** 2
                + self._raman_tensors[:, 1, 2] ** 2
            )
        ) / 2.0
        intensities = 45.0 * alpha_squared + 7.0 * gamma_squared

        if laser_correction:
            laser_wavenumber = 10000000 / laser_wavelength
            intensities *= get_laser_correction(
                self._phonon_wavenumbers, laser_wavenumber
            )
        if bose_einstein_correction:
            intensities *= get_bose_einstein_correction(
                self._phonon_wavenumbers, temperature
            )

        return self._phonon_wavenumbers, intensities
