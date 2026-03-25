# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

project = 'ModelArrayIO'
copyright = f'2017-{datetime.now(tz=UTC).strftime("%Y")}, PennLINC developers'
author = 'PennLINC developers'

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
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
sys.path.insert(0, os.path.abspath('../src'))

from github_link import make_linkcode_resolve  # noqa: E402

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    'modelarrayio',
    'https://github.com/pennlinc/ModelArrayIO/blob/{revision}/{package}/{path}#L{lineno}',
)


def _sync_overview_figure(app) -> None:
    """Ensure docs/_static/overview_structure.png exists for the figure in index.rst.

    README.rst references the same path for GitHub/PyPI. The canonical file may live
    at the repository root (historical layout); copy it into _static before the build
    when needed so Sphinx can embed it.
    """
    docs_dir = Path(app.srcdir).resolve()
    root_png = docs_dir.parent / 'overview_structure.png'
    static_png = docs_dir / '_static' / 'overview_structure.png'
    if root_png.is_file():
        static_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root_png, static_png)


def setup(app):
    app.connect('builder-inited', _sync_overview_figure)
