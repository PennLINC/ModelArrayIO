# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime

project = 'ModelArrayIO'
copyright = f'2017-{datetime.today().strftime("%Y")}, PennLINC developers'
author = 'PennLINC developers'

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
source_suffix = {
    '.md': 'markdown',
}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'ModelArrayIO'

myst_heading_slugs = True
suppress_warnings = ['image.not_readable']
