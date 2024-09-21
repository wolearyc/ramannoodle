Introduction
============

A chemical system's Raman spectrum reflects the frequencies and amplitudes at which the components of its **polarizability** -- a 3x3 tensor -- fluctuate due to thermal atomic motion. We must answer two questions to calculate a Raman spectrum of a collection of atoms:

1. How do the atoms vibrate at finite temperatures?
2. How does each vibration modulate polarizability?

To answer question (1), we often consider the system's vibrational normal modes, i.e., phonons in the case of periodic systems. However, in cases where atomic motion is appreciably anharmonic, the phonon picture misses important features. In these cases, we use molecular dynamics to understand, at least in a statistical sense, exactly how the atoms move as a function of time.

With the atomic dynamics in hand, answering question (2) is conceptually straightforward. When considering phonons, we calculate a **Raman tensor** for each phonon; mathematically speaking, a Raman tensor is the polarizability derivative in the direction of a phonon's displacement. We evaluate this derivative using central differences, which conventionally requires 2 polarizability calculations per phonon (though the exact number required depends on structural symmetries). When considering molecular dynamics, we calculate the polarizability at every timestep and assemble a **polarizability time series.**

Unfortunately, the need to calculate so many polarizabilities can make Raman spectrum calculations very computationally costly. These costs can quickly balloon, especially when treating large and/or complex systems. This ultimately makes conventional Raman spectrum calculations impractical for many of the most interesting and technologically relevant materials.

**Ramannoodle's primary purpose is to reduce the computational cost of first principles Raman calculations.** It accomplishes this by providing efficient polarizability models. For example, :class:`~ramannoodle.polarizability.art.ARTModel` and :class:`~ramannoodle.polarizability.interpolation.InterpolationModel` leverage structural symmetries to greatly reduce the number of required first principles polarizability calculations. We hope that current and future versions of this API will make computing Raman spectra simpler and, consequently, make Raman spectroscopy a more powerful characterization tool.

Installation
------------

Please see ramannoodle's `repo <https://github.com/wolearyc/ramannoodle>`_ for up-to-date installation instructions.

Citing
------

Please see ramannoodle's `repo <https://github.com/wolearyc/ramannoodle>`_ for up-to-date citation information.

Modules
--------

The following gives an overview of the modules available in ramannoodle.

1. :mod:`ramannoodle.io`
    .. automodule:: ramannoodle.io
        :synopsis:
        :no-index:

2. :mod:`ramannoodle.dynamics`
    .. automodule:: ramannoodle.dynamics
        :synopsis:
        :no-index:

3. :mod:`ramannoodle.structure`
    .. automodule:: ramannoodle.structure
        :synopsis:
        :no-index:

4. :mod:`ramannoodle.polarizability`
    .. automodule:: ramannoodle.polarizability
        :synopsis:
        :no-index:

5. :mod:`ramannoodle.spectrum`
    .. automodule:: ramannoodle.spectrum
        :synopsis:
        :no-index:

Basic Workflow
--------------

Ramannoodle's basic workflow is as follows:

1. Load in dynamics, for example phonons or a molecular dynamics trajectory.
2. Construct a polarizability model, which maps atomic positions to polarizabilities. We build-up (or train) this model by feeding in polarizability data calculated from first principles calculations.
3. Combine the polarizability model with the dynamics to compute a Raman spectrum.

Next, we will walk through a concrete example: :doc:`../notebooks/basics`
