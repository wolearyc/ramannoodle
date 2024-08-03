<div align="center">
  <img width="300" src="docs/logo_dark.png#gh-dark-mode-only">
  <img width="300" src="docs/logo.png#gh-light-mode-only">
</div>

-------
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) ![Tests](docs/tests-badge.svg) ![Coverage](docs/coverage-badge.svg) [![Documentation Status](https://readthedocs.org/projects/ramannoodle/badge/?version=latest)](https://ramannoodle.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)


**ramannoodle** helps you calculate Raman spectra from first-principles calculations.

> [!NOTE]
>  **This API is currently in alpha.**

ramannoodle is built from the ground up with the goals of being:

1. **EFFICIENT**

   ramannoodle provides `PolarizabilityModel`'s to reduce the required number of first-principles polarizability calculations.

2. **FLEXIBLE**

    ramannoodle provides a simple, object-oriented API that makes calculations a breeze while offering plenty of flexibility to carry out advanced analyses and add new functionality.

3. **TRANSPARENT**

    ramannoodle is designed according to the philosophy that the user should understand *exactly* what is being calculated, without hidden corrections or assumptions.

Supported DFT Software
----------------------
* VASP (currently under development)
* phonopy (planned)
