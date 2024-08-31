Units
=====

Ramannoodle uses the following units:

+--------------------------+---------------+
| Quantity                 | Units         |
+==========================+===============+
| Wavenumbers              | cm\ :sup:`-1` |
+--------------------------+---------------+
| Intensities              | Unitless      |
+--------------------------+---------------+
| Fractional coordinates   | Unitless      |
+--------------------------+---------------+
| Cartesian coordinates    | Å             |
+--------------------------+---------------+
| Polarizability\ :sup:`1` | Unitless      |
+--------------------------+---------------+
| Temperature              | K             |
+--------------------------+---------------+
| Wavelength               | nm            |
+--------------------------+---------------+
| Time                     | fs            |
+--------------------------+---------------+

\ :sup:`1` Dielectric tensor is used.

By default, Ramannoodle works with positions, displacements, and directions in fractional coordinates. Some parameters are expressed in Cartesian coordinates; these parameters always have the prefix ``cart_`` appended:

.. code-block:: python

      some_function(..., positions, ...) # positions is in fractional coordinates
      some_other_function(..., cart_positions, ...) # cart_positions is in Cartesian coordinates
