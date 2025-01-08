import os
from os.path import relpath, dirname
from pathlib import Path
import re
import sys
import warnings
from datetime import date

sys.path.insert(0, str(Path('..', '..', 'src', 'python').resolve()))


import spatialize

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spatialize'
copyright = f'2024-{date.today().year}, ALGES Lab'
author = 'ALGES Lab'
version = re.sub(r'\.dev.*$', r'.dev', spatialize.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.linkcode",
    'sphinx_math_dollar',
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_mock_imports = ['libspatialize', 'cv2']
autodoc_default_options = {
    'members': True
}
autodoc_docstring_signature = True

# -----------------------------------------------------------------------------
# numpydoc
# -----------------------------------------------------------------------------
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_attributes_as_param_list = False
numpydoc_class_members_toctree = False

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

# autosummary_generate = True

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

pygments_style = "sphinx"

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
  "show_nav_level": 2
}
