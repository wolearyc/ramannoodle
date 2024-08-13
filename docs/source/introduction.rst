Introduction
============

A chemical system's Raman spectrum reflects the frequencies and amplitudes at which the components of its **polarizability** -- a 3x3 tensor -- fluctuate due to thermal atomic motion. Therefore, we must answer two questions to calculate a Raman spectrum:

1. How do the atoms "jiggle" at finite temperatures?
2. How does each jiggle modulate polarizability?

To answer question (1), we often simply consider the system's vibrational normal modes, i.e., phonons in the case of periodic systems. However, in cases where atomic motions are strongly anharmonic, the phonon picture misses important features; in these cases, we use molecular dynamics to understand, at least in a statistical sense, exactly how the atoms move as a function of time.

With the atomic dynamics in hand, answering question (2), is, at least conceptually, quite straightforward. When considering phonons, we  calculate a **Raman tensor** for each phonon; mathematically speaking, this is a directional derivative of polarizability in the direction of that a phonon displacement. We evaluate this derivative using central differences, which conventionally requires ~2 polarizability calculations per phonon. When using molecular dynamics, our task is even simpler; we calculate the polarizability at every timestep and assemble a **polarizability time series.**

Unfortunately, the need to calculate so many polarizabilities can make Raman spectrum calculations rather computationally costly. These costs can quickly balloon, especially when treating large and/or complex systems. These costs ultimately make Raman spectra calculations impractical for many of the most interesting and technologically relevant materials.

**Ramannoodle** was designed to reduce the cost to calculate Raman spectra from first principles. It does this by providing efficient :class:`~ramannoodle.polarizability.abstract.PolarizabilityModel`s, which leverage structural symmetries to greatly reduce the number of required first principles polarizability calculations. The plan is to extend this API with additional models and capabilities that will make computing Raman spectra a breeze and, consequently, make Raman spectroscopy a more powerful characterization tool.

Installation
------------

Ramannoodle can be installed - as as standard for Python packages - with pip:

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

Ramannoodle's basic workflow is as follows:

1. We load in dynamics, for example phonons or a molecular dynamics trajectory.
2. We construct a polarizability model, which maps atomic positions to polarizabilities. We build-up (or train) this model by feeding in polarizability data calculated from first principles calculations.
3. We combine the polarizability model with the dynamics to compute a Raman spectrum.

Next, we will walk through a concrete example: :doc:`../notebooks/Basic tutorial`
