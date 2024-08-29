Interfacing with first-principles software
==========================================

Ramannoodle includes routines for reading and writing files used by quantum chemistry software. Currently, ramannoodle only supports VASP. However, we hope to include interfaces with other codes in future versions.

IO
---

IO operations are implemented in ramannoodle as functions. These are organized into packages and modules under :mod:`ramannoodle.io`. For example, VASP POSCAR IO functions are contained in :mod:`ramannoodle.io.vasp.poscar` while OUTCAR files can be read with functions in :mod:`ramannoodle.io.vasp.outcar`. Using this structure, files of various types can be interacted with in an intuitive and readable way:

.. code-block:: python

      import ramannoodle.io.vasp as vasp_io

      phonons = vasp_io.outcar.read_phonons(...)
      vasp_io.poscar.write_structure(...)

Ramannoodle also includes generic versions of IO functions in :mod:`ramannoodle.io.generic`. These functions use parameters to specify the desired file format:

.. code-block:: python

      import ramannoodle.io.generic as generic_io

      phonons = generic_io.read_phonons(..., file_format = "outcar")
      generic_io.write_structure(..., file_format = "poscar")

These generic functions are less flexible than those first mentioned, and therefore these generic functions are best used only when necessary. One such case is loading files directly into polarizability models:

.. code-block:: python

      import ramannoodle.io.generic as generic_io

      model = InterpolationModel(...)
      model.add_dof_from_files(..., file_format = "outcar")

:meth:`.InterpolationModel.add_dof_from_files` and other methods like it rely on these generic methods, as apparent from the ``file_format`` argument.

.. _Supported formats:

Supported formats
-----------------

The following table reviews which file types and properties are currently supported by ramannoodle's IO functions:

+---------------------------------+------------------------------+-------------------+----------------------+----------------------+----------------+
| File format (``file_format``)   | :class:`.ReferenceStructure` | :class:`.Phonons` | :class:`.Trajectory` | positions            | polarizability |
+=================================+==============================+===================+======================+======================+================+
| POSCAR (``"poscar"``)           | read/write                   |                   |                      | read/write           |                |
+---------------------------------+------------------------------+-------------------+----------------------+----------------------+----------------+
| OUTCAR (``"outcar"``)           | read\ :sup:`1`               | read              | read                 | read\ :sup:`1`       | read           |
+---------------------------------+------------------------------+-------------------+----------------------+----------------------+----------------+
| XDATCAR (``"xdatcar"``)         | read\ :sup:`1`/write         |                   | read\ :sup:`2`/write | read\ :sup:`1`/write |                |
+---------------------------------+------------------------------+-------------------+----------------------+----------------------+----------------+
| vasprun.xml (``"vasprun.xml"``) | read\ :sup:`1`               | read              | read                 | read\ :sup:`1`       | read           |
+---------------------------------+------------------------------+-------------------+----------------------+----------------------+----------------+

:sup:`1` Uses initial structure.
:sup:`2` Not available in ``generic_io``
