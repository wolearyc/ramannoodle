"""
Sample includes:

from dynamics import Phonons
from dynamics import MDTrajectory

from spectrum import RamanSpectrum
from spectrum import VibrationalSpectrum

from spectrometer import RamanSpectrometer
from spectrometer import VibrationalSpectrometer

from spectrum.phonons import PhononRamanSpectrum
from spectrum.md import MDRamanSpectrum

from polarizability import PolarizabilityModel

from polarizability.taylor import LinearPolarizabilityModel


Workflow:

From your simulations:
- Create a Dynamics object.
- Create a PolarizabilityModel object.

Run a virtual experiment
- Specify SpectrometerSettings (angle, polycrystalline)
- dynamics.get_spectrum(PolarizabilityModel, SpectrometerSettings) -> RamanSpectrum

Extract Relevant information from RamanSpectrum
- Spectrum.get_intensities(temperature, correction factors, smearing, etc.)
- Spectrum.get_wavenumbers()
-

For now, I will not implement SpectrometerSettings. This can wait.

"""
