<div align="center">
  <img width="200" src="docs/source/_static/logo_dark.png#gh-dark-mode-only">
  <img width="200" src="docs/source/_static/logo.png#gh-light-mode-only">
</div>

-------
![PyPI - Version](https://img.shields.io/pypi/v/ramannoodle?color=dark%20green) [![python](https://img.shields.io/badge/python-3.10|3.11|3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Tests](docs/tests-badge.svg) ![Coverage](docs/coverage-badge.svg) [![Documentation Status](https://readthedocs.org/projects/ramannoodle/badge/?version=latest)](https://ramannoodle.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

## About

**Ramannoodle** is a Python API for efficiently calculating Raman spectra from first principles calculations. Ramannoodle supports molecular-dynamics- and phonon-based Raman calculations and includes interfaces with VASP.

Ramannoodle aims to be:

1. **EFFICIENT**

   Ramannoodle provides `PolarizabilityModel`'s to reduce the required number of first-principles polarizability calculations.

2. **FLEXIBLE**

    Ramannoodle provides a simple, object-oriented API that makes calculations a breeze while offering plenty of flexibility to carry out advanced analyses and add new functionality.

3. **TRANSPARENT**

    Ramannoodle is designed to give the user a good understanding of what is being calculated at varying levels of abstraction.

Ramannoodle includes interfaces with:

* VASP
* phonopy (planned)

## Installation

Ramannoodle can be installed via pip:

```
$ pip install ramannoodle
```

Due to idiosyncrasies with PyTorch's build system, installing ramannoodle's machine learning modules is slightly more involved. First, PyTorch must be installed ([pip commands](https://pytorch.org/get-started/locally/)). Then, corresponding torch-scatter and torch-sparse packages must be installed. Finally, Ramannoodle can then be installed with the appropriate options.

For example, installation on a Linux system using PyTorch 2.4.1 (cpu implementation) is done as follows:

```
$ pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu
$ pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
$ pip install ramannoodle[torch]
```

## Documentation

[https://ramannoodle.readthedocs.io/](https://ramannoodle.readthedocs.io/)

## Contributing

Contributions in the form of bug reports, feature suggestions, and pull requests are always welcome! Those contributing code should check out the [dev guide](https://ramannoodle.readthedocs.io/en/latest/development.html).

## Citing

coming soon...

## Future releases

* **0.4.0** | ML polarizability models
* **0.5.0** | Advanced spectra analyses
* **1.0.0** | Official release
