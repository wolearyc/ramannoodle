Usage
=====

Installation
------------

Coming soon.

General Usage
----------------------------------

Although dielectric models can be generated manually, we recommend using ramannoodle's build-in routines to set up and process calculations. Building the dielectric model for a specific site is done in three steps:

   1. Set up displaced structures using ramannoodle :py:func:`ramannoodle.setup_dielectric_displacement`
   2. Run dielectric calculations on these displaced structures
   3. Load in dielectric tensors from these calculations with :py:func:`ramannoodle.get_dielectric_model`

Setting up dielectric displacement calculations
-----------------------------------------------

For a given structure, you will want to make a new directory for a single structure. Within this directory, make a new directory called ``ref``. Within this directory, set up a VASP dielectric tensor calculation for the system's reference structure. In most cases, this will be a high symmetry structure resulting from structural optimization. This could also be the average structure predicted by molecular dynamics, though without symmetrization the benefits of ramannoodle's approach may be lost. 

.. note::
   After the dielectric tensor calculation completes, one must exact the dielectric tensor data from the OUTCAR and move it into a file named ``dielectric_tensor`` located in the calculation directory. This should be done using the following command:

   .. code-block:: console
   
      $ grep 'MACROSCOPIC STATIC DIELECTRIC TENSOR' /path/to/OUTCAR -A 5 >> /calculation/directory/dielectric_tensor

   We recommend simply appending this grep command to the end of your batch script.

After the calculation ends, your directory structure should be roughly as follows:

..  code-block:: none
    
    dielectric_root_directory
    └── ref
        ├── INCAR
        ├── KPOINTS
        ├── ...
        └── dielectric_tensor

Now, we can set up additional calculations for all crystallographically distinct atoms and displacement directions. Fully generated, the directory structure will be 

..  code-block:: none
    
    dielectric_root_directory
    ├── Atom1
        ├── x
            ├── m0.1
            ├── ...
            └── 0.1
        ├── ...
        └── z
    ├── Atom2
    ├── Atom3
    ├── ...
    └── ref

Here, ``Atom1`` etc. serve as placemarkers. ``Atom1/x/m0.1`` will contain a dielectric calculation. It is important that this calculation be done identically to that done in ``ref``. Therefore, we recommend building a script to copy input files from ``ref`` into the appropriate subdirectory(ies) before running the calculation.

.. note::
   
   Raman noodle contains functions to help determine which atoms and directions are crystalographically equivalent. Documentation on this is coming soon...

To set up a dielectric displacement calculation, apply :py:func:`ramannoodle.setup_dielectric_displacement`

   .. autofunction:: ramannoodle.setup_dielectric_displacement

Within the context of the directory structure mentioned above, ``root_dir`` would be the path to ``dielectric_root_directory``. This function uses the POSCAR within ``ref`` as it's reference structure. It then creates the appropriate directories and writes POSCARs (with a specific atom displaced in some direction) into the appropriate directories. 

It is then up to the user to run dielectric calculations on each one of these POSCARs. We recommend at least partially automating this task. 


Constructing a system's dielectric model
----------------------------------------

A system's total dielectric model is made up essentially of a list of site-specific dielectric models. We construct the total dielectric model using :py:class:`ramannoodle.SymmetricDielectricModel`.

.. code-block:: python
    
   # in the case of our example, this would point to 'dielectric_root_directory'
   root_dir = '/path/to/root/dir'

   model = SymmetricDielectricModel(f'{root_dir}/ref/POSCAR')

We then load in the relevant dielectric models and add them to ``model``:

..  code-block:: python
    
   model1x = get_dielectric_model(root_dir = root_dir,
                                  atom_number = 1, 
                                  directory_name = 'x')
   model1y = get_dielectric_model(root_dir = root_dir,
                                   atom_number = 1, 
                                   directory_name = 'y')
   ...
   model20z = get_dielectric_model(root_dir = root_dir,
                                   atom_number = 20, 
                                   directory_name = 'z')

   for dmodel in [model1x, model1y, ..., model20z]:
      model.add_dielectric_model(dmodel)

Once all the models are added, the total dielectric model is complete and can be used to calculate a Raman spectrum.

Calculating a Raman spectrum
----------------------------
With a molecular dynamics trajectory and a dielectric model, calculating a raman specturm is fairly easy.

..  code-block:: python
    
   model = SymmetricDielectricModel(...)
   # add relevant DielectricModels to model

   data = Trajectory(...)

   wavenumbers, intensities = get_raman(data, 
                                        model.dielectric_models, 
                                        smearing = 5)

Congrats! You have calculated a Raman spectrum.

Molecular dynamics recipes
--------------------------

Coming soon...


