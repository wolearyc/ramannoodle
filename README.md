<div align="center">
  <img width="200" src="docs/source/_static/logo_dark.png#gh-dark-mode-only">
  <img width="200" src="docs/source/_static/logo.png#gh-light-mode-only">
</div>

-------
![PyPI - Version](https://img.shields.io/pypi/v/ramannoodle?color=dark%20green) [![python](https://img.shields.io/badge/python-3.10|3.11|3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Tests](docs/tests-badge.svg) ![Coverage](docs/coverage-badge.svg) [![Documentation Status](https://readthedocs.org/projects/ramannoodle/badge/?version=latest)](https://ramannoodle.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

## About

**Ramannoodle** is a Python API for efficiently calculating Raman spectra from first principles calculations. Ramannoodle supports molecular-dynamics- and phonon-based Raman calculations. It includes interfaces with VASP but can easily be used with other codes using IO from external libraries, such as [pymatgen](https://pymatgen.org/) or [ase](https://wiki.fysik.dtu.dk/ase/).

Ramannoodle aims to be:

1. **EFFICIENT**

   Ramannoodle provides `PolarizabilityModel`'s to reduce the required number of first-principles polarizability calculations.

2. **FLEXIBLE**

    Ramannoodle provides a simple, object-oriented API that makes calculations a breeze while offering plenty of flexibility to carry out advanced analyses and add new functionality.

3. **TRANSPARENT**

    Ramannoodle is designed to give the user a good understanding of what is being calculated at varying levels of abstraction.

## Installation

The base version of ramannoodle can be installed with pip:

```
$ pip install ramannoodle
```

Ramannoodle's machine learning modules are implemented with PyTorch. To use these modules:
1. Install [PyTorch](https://pytorch.org/get-started/locally/).
2. Install [torch-scatter](https://pypi.org/project/torch-scatter/) and [torch-sparse](https://pypi.org/project/torch-sparse/) corresponding to the PyTorch version/implementation.
3.  Install ramannoodle using the `torch` options group.

For example, installation on a Linux system using PyTorch 2.4.1 (cpu implementation) is done as follows:

```
$ pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu
$ pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
$ pip install ramannoodle[torch]
```

Ramannoodle includes interfaces with [pymatgen](https://pymatgen.org/). To use these interfaces, ramannoodle should be installed with the `[pymatgen]` option:

```
$ pip install ramannoodle[pymatgen]
```

## Tutorials and docs

[https://ramannoodle.readthedocs.io/](https://ramannoodle.readthedocs.io/)

## Contributing

Contributions in the form of bug reports, feature suggestions, and pull requests are always welcome! Those contributing code should check out the [dev guide](https://ramannoodle.readthedocs.io/en/latest/development.html).

## Citing

To acknowledge use of ramannoodle, please cite

>> **Rapid Characterization of Point Defects in Solid-State Ion Conductors Using Raman Spectroscopy, Machine-Learning Force Fields, and Atomic Raman Tensors** <br>
 W. O’Leary, M. Grumet, W. Kaiser, T. Bučko, J.L.M. Rupp, D.A. Egger <br>
 Journal of the American Chemical Society (2024) <br>
 doi: [10.1021/jacs.4c07812](https://pubs.acs.org/doi/10.1021/jacs.4c07812)
