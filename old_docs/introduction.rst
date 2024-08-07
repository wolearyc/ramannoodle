Introduction
============

A chemical system's Raman spectrum reflects the frequencies and amplitudes at which the components of its **polarizability** -- a 3x3 tensor -- fluctuate due to thermal atomic motion. Therefore, we must answer two questions to calculate a Raman spectrum:

1. How do the atoms "jiggle" at finite temperatures?
2. How does polarizability depend on these jiggles?

To answer question (1), we often simply consider the system's vibrational normal modes, i.e., phonons in the case of periodic systems. However, in cases where atomic motions are strongly anharmonic, phonons provide a rather incomplete picture; in these cases, we use molecular dynamics to understand, at least in a statistical sense, exactly how the atoms move as a function of time.

With the atomic motions in hand, answering question (2), is, at least conceptually, quite straightforward. When considering phonons, we  calculate a **Raman tensor** for each phonon; mathematically speaking, this is a directional derivative of polarizability in the direction of that a phonon displacement. When using molecular dynamics, our task is even simpler; we simply calculate the polarizability at every timestep and assemble a **polarizability time series.** Unfortunately, calculation of polarizabilities comes with a high computational cost. This costs can quickly balloon, especially when treating large and/or complex systems. These costs ultimately make Raman spectra calculations impractical for many interesting and relevant systems.

ramannoodle was designed to reduce the cost to calculate Raman spectra from first principles. It does this by providing an efficient :class:`~ramannoodle.polarizability.interpolation.InterpolationPolarizabilityModel`, which leverages structural symmetries to greatly reduce the number of required first principles polarizability calculations. The plan is to extend this API with additional models and capabilities that will make computing Raman spectra (and ultimately interpreting experimental Raman spectra) a breeze.

Installation
------------

ramannoodle can be installed, like nearly all Python packages, with pip:

.. code-block:: console

      $ pip install ramannoodle

So long as your Python environment is configured correctly, you should be good to go:

.. code-block:: python

    import ramannoodle
    # ...

Modules
--------

The following gives an overview of the modules available in ramannoodle.

1. :mod:`ramannoodle.io`
    .. automodule:: ramannoodle.io
        :synopsis:

2. :mod:`ramannoodle.dynamics`
    .. automodule:: ramannoodle.dynamics
        :synopsis:

3. :mod:`ramannoodle.polarizability`
    .. automodule:: ramannoodle.polarizability
        :synopsis:

4. :mod:`ramannoodle.spectrum`
    .. automodule:: ramannoodle.spectrum
        :synopsis:

Basic Workflow
--------------

To give a bird-eye view of ramannoodle's workflow, this section will avoid too many specifics. A complete tutorial can be found in the next section.

First, ramannoodle needs to be fed relevant information from first principles calculations. Typically, the first data to be read are the atom dynamics.

.. code-block:: python

    from ramannoodle.io import read_phonons
    from ramannoodle.dynamics import Phonons

    phonons: Phonons = read_phonons(
        "/path/to/phonon/file",
    )

With the phonons in hand, we now need a suitable :class:`~ramannoodle.polarizability.PolarizabilityModel`. In this example, we will choose the simplest such model: the :class:`~ramannoodle.polarizability.interpolation.InterpolationPolarizabilityModel`. This model uses the symmetries of the system to significantly reduce the number of required polarizability calculations. We therefore load in the symmetries of a minimized structure for our system (along with the polarizability of the minimized structure), then use this information to initialize the model.

.. code-block:: python

    from ramannoodle.io import (
        read_structural_symmetry, read_positions_and_polarizability
    )

    from ramannoodle.symmetry import StructuralSymmetry
    from ramannoodle.polarizability.interpolation import InterpolationPolarizabilityModel

    symmetry: StructuralSymmetry = read_structural_symmetry(
        "/path/to/minimized/structure/file",
    )
    _, equil_polarizability = read_positions_and_polarizability(
        "/path/to/minimized/polarizability/file",
    )
    model = InterpolationPolarizabilityModel(symmetry, equil_polarizability)

We then feed in calculations, typically a collection of polarizabilities of distorted structures, to construct the model. These details will be left out for now.

.. code-block:: python

    # train the model

Once the model is complete, we are ready to predict a spectrum!

.. code-block:: python

    spectrum = phonons.get_raman_spectrum(model)
    wavenumbers, intensities = spectrum.measure(...)

And we're done!

To summarize, ramannoodle's basic workflow is as follows:

1. Load in dynamics, for example phonons or a molecular dynamics trajectory.
2. Construct a polarizability model, which maps atomic positions to polarizabilities.
3. Combine the polarizability model with the dynamics to compute a raman spectrum.
