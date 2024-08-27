"""Testing for spectra."""

import numpy as np
from numpy.typing import NDArray
import pytest

from ramannoodle.spectrum.spectrum_utils import calc_signal_spectrum


@pytest.mark.parametrize(
    "signal, sampling_rate",
    [
        (np.random.random(40), 1.0),
        (np.random.random(51), 1.0),
    ],
)
def test_calc_signal_spectrum(
    signal: NDArray[np.float64],
    sampling_rate: float,
) -> None:
    """Test calc_signal_spectrum."""
    wavenumbers, intensities = calc_signal_spectrum(signal, sampling_rate)
    assert wavenumbers.shape == (int(np.ceil(len(signal) / 2)),)
    assert intensities.shape == wavenumbers.shape
