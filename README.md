<div align="center">
  <img width="200" src="docs/source/_static/logo_dark.png#gh-dark-mode-only">
  <img width="200" src="docs/source/_static/logo.png#gh-light-mode-only">
</div>

-------
[![python](https://img.shields.io/badge/Python-3.10|3.11|3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Tests](docs/tests-badge.svg) ![Coverage](docs/coverage-badge.svg) [![Documentation Status](https://readthedocs.org/projects/ramannoodle/badge/?version=latest)](https://ramannoodle.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

## About

**ramannoodle** is a Python API that helps you calculate Raman spectra from first-principles calculations.

> [!NOTE]
>  **ramannoodle is currently in alpha.**

ramannoodle is built from the ground up with the goals of being:

1. **EFFICIENT**

   ramannoodle provides `PolarizabilityModel`'s to reduce the required number of first-principles polarizability calculations.

2. **FLEXIBLE**

    ramannoodle provides a simple, object-oriented API that makes calculations a breeze while offering plenty of flexibility to carry out advanced analyses and add new functionality.

3. **TRANSPARENT**

    ramannoodle is designed according to the philosophy that the user should understand *exactly* what is being calculated, without hidden corrections or assumptions.

**ramannoodle interfaces with...**

* VASP (currently under development)
* phonopy (planned)

## Installation

ramannoodle can be installed via pip:

`
pip install ramannoodle
`

## Documentation

[https://ramannoodle.readthedocs.io/](https://ramannoodle.readthedocs.io/)

## Contributing

Contributions in the form of bug reports, feature suggestions, and pull requests are always welcome!

## Citing

coming soon...

## Roadmap

Current release: 0.2.0

Future releases:

**0.3.0**
* Add support for molecular dynamics
* Add IO support for Phonopy

**1.0.0**
* Official release

**1.1.0**
* ML polarizability models
