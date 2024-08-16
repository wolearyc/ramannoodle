Units
=====

Ramannoodle uses the following units:

+------------------------+--------------------+
| Quantity               | Units              |
+========================+====================+
| Wavenumbers            | cm\ :sup:`-1`      |
+------------------------+--------------------+
| Intensities            | Arbitrary units    |
+------------------------+--------------------+
| Fractional coordinates | Unitless           |
+------------------------+--------------------+
| Polarizability         | Unitless\ :sup:`*` |
+------------------------+--------------------+
| Cartesian coordinates  | Å                  |
+------------------------+--------------------+
| Temperature            | K                  |
+------------------------+--------------------+
| Wavelength             | nm                 |
+------------------------+--------------------+

\ :sup:`*` Dielectric tensor is used.

Ramannoodle almost always expresses positions, displacements, and directions in fractional coordinates. Those parameters that are expressed in Cartesian coordinates always have the prefix ``cart_`` appended

.. code-block:: python

      some_function(..., positions, ...) # positions is in fractional coordinates
      some_other_function(..., cart_positions, ...) # cart_positions is in Cartesian coordinates
