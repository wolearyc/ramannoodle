"""Sphinx conf.py."""
# pylint: skip-file
# flake-8: noqa
# cspell: disable
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

project = 'ramannoodle'
copyright = "2024, Willis O'Leary"
author = "Willis O'Leary"
release = 'v0.1.1-alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting'
]
autodoc_typehints = 'description'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('../../'))
