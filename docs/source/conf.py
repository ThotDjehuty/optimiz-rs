# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../python'))

# -- Project information -----------------------------------------------------
project = 'Optimiz-rs'
copyright = '2026, HFThot Research Lab'
author = 'HFThot Research Lab'
release = '0.3.0'
version = '0.3.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'  # Modern, clean theme
html_static_path = ['_static']
html_title = 'Optimiz-rs Documentation'
html_short_title = 'Optimiz-rs'
html_logo = 'logo_optimizrs.png'
html_favicon = 'logo_optimizrs.png'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#f97316",
        "color-brand-content": "#f97316",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fb923c",
        "color-brand-content": "#fb923c",
    },
}

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MyST parser configuration for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
