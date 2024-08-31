"""Raman spectrum classes and utilities."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import get_type_error, verify_ndarray_shape
from ramannoodle.globals import BOLTZMANN_CONSTANT
from ramannoodle.spectrum.abstract import RamanSpectrum

from ramannoodle.spectrum.spectrum_utils import calc_signal_spectrum


def get_bose_einstein_correction(
    wavenumbers: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    r"""Calculate Bose-Einstein spectral correction.

    Parameters
    ----------
    wavenumbers
        | (cm\ :sup:`-1`) 1D array with shape (M,).
    temperature
        | (K)

    Returns
    -------
    :
        1D array with shape (M,).

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
        | (cm\ :sup:`-1`) 1D array with shape (M,).
    laser_wavenumber
        | (cm\ :sup:`-1`)

    Returns
    -------
    :
        1D array with shape (M,).

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


class PhononRamanSpectrum(RamanSpectrum):
    r"""Phonon-based first-order Raman spectrum.

    The spectrum is specified by a list of phonon wavenumbers and corresponding Raman
    tensors.

    Parameters
    ----------
    phonon_wavenumbers
        | (cm\ :sup:`-1`) 1D array with shape (M,) where M is the number of phonons.
    raman_tensors
        | 3D array with shape (M,3,3).

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
        r"""Get (a copy of) phonon wavenumbers.

        Returns
        -------
        :
            (cm\ :sup:`-1`) 1D array with shape (M,) where M is the number of phonons.
        """
        return self._phonon_wavenumbers.copy()

    @property
    def raman_tensors(self) -> NDArray[np.float64]:
        """Get (a copy of) Raman tensors.

        Returns
        -------
        :
            3D array with shape (M,3,3) where M is the number of phonons.
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
            Supports ``"polycrystalline"``.

            Future versions will support arbitrary orientations.
        laser_correction
            | Whether to apply laser-wavelength-dependent intensity correction.
        laser_wavelength
            | (nm) Ignored if ``laser_correction == False``.
        bose_einstein_correction
            | Whether to apply temperature-dependent Bose Einstein correction.
        temperature
            | (K) Ignored if ``bose_einstein_correction == False``.

        Returns
        -------
        :
            2-tuple:
                0. | wavenumbers --
                   | (cm\ :sup:`-1`) 1D array with shape (M,).
                #. | intensities --
                   | (arbitrary units) 1D array with shape (M,).

        Raises
        ------
        NotImplementedError
            Raised when any orientation besides ``"polycrystalline"`` is supplied.

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


class MDRamanSpectrum(RamanSpectrum):
    """Molecular-dynamics-based Raman spectrum.

    The spectrum is specified by a polarizability time series (``polarizability_ts``)
    and a timestep.

    Parameters
    ----------
    polarizability_ts
        | 3D array with shape (S,3,3) where S is the number of configurations.
    timestep
        | (fs)

    """

    def __init__(self, polarizability_ts: NDArray[np.float64], timestep: float):
        verify_ndarray_shape("polarizability_ts", polarizability_ts, (None, 3, 3))

        self._polarizability_ts = polarizability_ts
        self._timestep = timestep

    @property
    def polarizability_ts(self) -> NDArray[np.float64]:
        """Get (a copy) of polarizability time series.

        Returns
        -------
        :
            3D array with shape (S,3,3) where S is the number of configurations.
        """
        return self._polarizability_ts

    @property
    def timestep(self) -> float:
        """Get timestep.

        Returns
        -------
        :
            (fs)
        """
        return self._timestep

    def measure(  # pylint: disable=too-many-arguments
        self,
        orientation: str | NDArray[np.float64] = "polycrystalline",
        laser_correction: bool = False,
        laser_wavelength: float = 522,
        bose_einstein_correction: bool = False,
        temperature: float = 300,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Calculate and return a raw Raman spectrum.

        .. note:: Raw MD-derived Raman spectra will typically need to be smoothed to be
                  visualized effectively. See :func:`.spectrum_utils.convolve_spectrum`.

        Parameters
        ----------
        orientation
            Supports ``"polycrystalline"``.

            Future versions will support arbitrary orientations.
        laser_correction
            | Whether to apply laser-wavelength-dependent intensity correction.
        laser_wavelength
            | (nm) Ignored if ``laser_correction == False``.
        bose_einstein_correction
            | Whether to apply temperature-dependent Bose Einstein correction.
        temperature
            | (K) Ignored if ``bose_einstein_correction == False``.

        Returns
        -------
        :
            2-tuple:
                0. | wavenumbers --
                   | (cm\ :sup:`-1`) 1D array with shape (ceiling(S / 2),) where S is
                   | the number of configurations.
                #. | intensities --
                   | (arbitrary units) 1D array with shape (ceiling(S / 2),).

        """
        if orientation != "polycrystalline":
            raise NotImplementedError(
                "only polycrystalline spectra are supported for now"
            )

        # "alpha dot" aka polarizability time derivative
        ad = np.diff(self._polarizability_ts, axis=0)
        timestep = self._timestep
        wavenumbers, _ = calc_signal_spectrum(ad[:, 0, 0], timestep)

        alpha2 = (1 / 9) * calc_signal_spectrum(
            ad[:, 0, 0] + ad[:, 1, 1] + ad[:, 2, 2], timestep
        )[1]
        gamma2 = (
            (1 / 2) * calc_signal_spectrum(ad[:, 0, 0] - ad[:, 1, 1], timestep)[1]
            + (1 / 2) * calc_signal_spectrum(ad[:, 1, 1] - ad[:, 2, 2], timestep)[1]
            + (1 / 2) * calc_signal_spectrum(ad[:, 2, 2] - ad[:, 0, 0], timestep)[1]
            + 3 * calc_signal_spectrum(ad[:, 0, 1], timestep)[1]
            + 3 * calc_signal_spectrum(ad[:, 1, 2], timestep)[1]
            + 3 * calc_signal_spectrum(ad[:, 0, 2], timestep)[1]
        )
        intensities = 45.0 * alpha2 + 7.0 * gamma2

        # Intensity @ 0 cm-1 is infinite, so we remove.
        intensities = intensities[1:]
        wavenumbers = wavenumbers[1:]

        if laser_correction:
            laser_wavenumber = 10000000 / laser_wavelength
            intensities *= get_laser_correction(wavenumbers, laser_wavenumber)
        if bose_einstein_correction:
            intensities *= get_bose_einstein_correction(wavenumbers, temperature)

        return wavenumbers, intensities
