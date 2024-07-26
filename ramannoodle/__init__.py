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

1. Create a Dynamics object from your simulations.

2. Create an InterpolationPolarizabilityModel from your simulations.
- Set up a Symmetry object from an OUTCAR
- Set up a model for that symmetry
- Insert DOF, a set of displacements and polarizabilities (e.g. +,-,
polarizability_plus,polarizability_minus)
    - Check that displacements given are colinear.
    - That that the new displacements are orthogonal to all existing displacements
    - If checks are successful:
        - Generate all symmetry equivalent displacements.
        - Do sanity check to ensure we're still totally orthogonal
        - add interpolation and normalized dof displacements

Run a virtual experiment
- Specify RamanSettings (angle, polycrystalline)
- dynamics.get_spectrum(PolarizabilityModel, SpectrometerSettings) -> RamanSpectrum

Extract Relevant information from RamanSpectrum
- Spectrum.get_intensities(temperature, correction factors, smearing, etc.)
- Spectrum.get_wavenumbers()
-

For now, I will not implement SpectrometerSettings. This can wait.

"""
