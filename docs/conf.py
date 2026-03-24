# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import UTC, datetime

project = 'ModelArrayIO'
copyright = f'2017-{datetime.now(tz=UTC).strftime("%Y")}, PennLINC developers'
author = 'PennLINC developers'

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinxarg.ext',
    'sphinx_copybutton',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
source_suffix = {'.rst': 'restructuredtext'}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'ModelArrayIO'

myst_heading_slugs = True
suppress_warnings = ['image.not_readable']

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath('sphinxext'))
sys.path.insert(0, os.path.abspath('../modelarrayio'))

from github_link import make_linkcode_resolve  # noqa: E402

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    'modelarrayio',
    'https://github.com/pennlinc/ModelArrayIO/blob/{revision}/{package}/{path}#L{lineno}',
)
