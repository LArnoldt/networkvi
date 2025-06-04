import os
import sys

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'NetworkVI'
copyright = '2024, Lucas Arnoldt, Julius Upmeier zu Belzen, Luis Herrmann, Khue Nguyen, Fabian Theis, Benjamin Wild, Roland Eils'
author = 'Lucas Arnoldt, Julius Upmeier zu Belzen, Luis Herrmann, Khue Nguyen, Fabian Theis, Benjamin Wild, Roland Eils'

release = '1.0.0'
version = '1.0.0'

# -- General configuration

master_doc = 'index'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    "myst_parser",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'myst-nb',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme' #'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
    "colon_fence",
]

nb_execution_mode = "off" #"auto"
nb_execution_timeout = -1
