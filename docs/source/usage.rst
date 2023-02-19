Usage
=====

.. _installation:

Installation
------------

To use ramannoodle, first install it using pip:

.. code-block:: console

   (.venv) $ pip install ramannoodle

Overview
--------

To generate a Raman spectrum with ramannoodle, one needs:
#. A vasprun.xml file form a molecular dynamics calculation. This file holds the trajectory. 
#. A list of dielectric models. These are generated with python based on DFT-calculated dielectric tensors.


Molecular dynamics recipes
--------------------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

