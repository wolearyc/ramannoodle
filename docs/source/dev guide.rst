Dev Guide
=========

Contributions to ramannoodle - in the form of bug reports, feature suggestions, pull requests, etc. -  are always welcome! We are particularly interested in adding new interfaces for first-principles codes as well as new polarizability models. This document provides a few development guidelines. Please reach out on Github with any questions or if you want to collaborate :)

Workflow
--------

Ramannoodle is hosted on Github:

`https://github.com/wolearyc/ramannoodle <https://github.com/wolearyc/ramannoodle>`_

The project uses `pre-commit <https://pre-commit.com/>`_ with a range of hooks that help ensure high code quality and consistency. You are strongly encouraged to use it frequently, as all pull requests need to pass CI pre-commit before merge! Pre-commit can be installed with pip

.. code-block:: console

      $ pip install pre-commit

and can be run from the repository's root directory with

.. code-block:: console

      $ pre-commit run --all

Ramannoodle includes a test suite that uses the `pytest <https://docs.pytest.org/en/stable/>`_ framework. Pull requests must pass all CI tests in order to be merged. New tests should be developed for any new functionality.

Ramannoodle's documentation is built with `Sphinx <https://www.sphinx-doc.org/en/master/>`_. The documentation can be built by running

.. code-block:: console

      $ make html

from within the docs directory. The resulting html is available in docs/build/html.

Guidelines
----------

In no particular order, here are some guidelines that are followed throughout ramannoodle:

* Since `mypy <https://mypy-lang.org/>`_ is used, type hints are mandatory.

* Whenever possible, use the EAFP (as opposed to LBYL) principle when raising exceptions.

* Use numpy-style docstrings.

* All public-facing functions should raise descriptive TypeError and ValueError exceptions when invalid arguments are provided. These sorts of exceptions should not be documented in the docstring.

* All array arguments containing floats should be numpy arrays. This should be enforced, when appropriate, through exceptions.

* Docstring descriptions for array arguments should provide a dimension and shape. Uppercase letters can be used for cases where shape is variable. For example, "4D array with shape (M,N,3) where M is ... and N is ...".

* For all coordinates, vectors, displacements, etc, ramannoodle works in fractional coordinates. Variables and arguments in cartesian coordinates will also have  "cart\_" appended.

* Use classes widely. Sometimes, a regular function is all that is needed!

* With IO functions, ramannoodle attempts to strike a balance between simplicity and flexibility. ``import ramannoodle.io.generic`` provides access to generic file readers and writers for a variety of file formats. These generic routines are rather inflexible but are necessary for certain functionality. Users are strongly encouraged to use functions contained in the code-specific IO packages, such as ``import ramannoodle.io.vasp.poscar``.
