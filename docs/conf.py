# Configuration file for Sphinx documentation builder
# This is needed for autodoc to work with MyST

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'fempack'
copyright = '2025, Diego T. Volpatto'
author = 'Diego T. Volpatto'

# General configuration
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

# MyST configuration
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'colon_fence',
    'substitution',
    'eval-rst',
]

# Napoleon settings for NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
