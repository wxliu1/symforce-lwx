# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add symforce to python path
package_path = os.path.abspath("..")
sys.path.insert(0, package_path)
os.environ["PYTHONPATH"] = package_path + ":" + os.environ.get("PYTHONPATH", "")

# -- Project information -----------------------------------------------------

project = "symforce"
copyright = "2022, Skydio, Inc"
author = "Skydio"

# The short X.Y version
from symforce import __version__ as version

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "breathe",
    "myst_parser",
]

autodoc_typehints = "description"

# This doesn't seem to fix things, but would be good to fix - currently you have to manually write
# anchor links in html
myst_heading_anchors = 3

# Enable $$ $$ and $ $ math delimiters in MyST markdown
myst_enable_extensions = ["dollarmath"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Include special methods
napoleon_include_special_with_doc = True

# Make "Returns" a params-style section, so we can document multiple returns
napoleon_custom_sections = [("Returns", "params_style")]

autodoc_default_options = {
    # Order by definition in source rather than alphabetical
    "member-order": "bysource",
    # Include undocumented members
    "undoc-members": True,
}
# Order by definition in source rather than alphabetical
autodoc_member_order = "bysource"

# Show the class docstring on the class, not the init
autoclass_content = "class"

# Don't prefix full module path on definitions
add_module_names = False

# Don't add copybuttons to notebook output blocks
copybutton_selector = "div:not(.output_area) > div.highlight > pre"

# Skip symforce for module index
modindex_common_prefix = ["symforce."]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# This requires Sphinx 5.2+
toc_object_entries_show_parents = "hide"

# Add links to the Python, numpy, sympy, and scipy docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Image for OpenGraph
ogp_image = "https://symforce.org/docs/static/images/symforce_banner.png"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["docs/source/api/modules", "**/build", "**/.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "dark_logo": "images/symforce vertical.svg",
    "light_logo": "images/symforce vertical dark.svg",
    "sidebar_hide_name": True,
}

html_title = "SymForce"
html_favicon = "static/favicon.ico"
html_css_files = ["css/custom.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "symforcedoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements: dict = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "symforce.tex", "symforce Documentation", "Hayk Martiros", "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "symforce", "symforce Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "symforce",
        "symforce Documentation",
        author,
        "symforce",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Breathe configuration ---------------------------------------------------

breathe_projects = {
    "api-cpp": "../build/doxygen-cpp/xml",
    "api-gen-cpp": "../build/doxygen-gen-cpp/xml",
}
breathe_default_project = "api-cpp"
breathe_default_members = ("members", "undoc-members")
breathe_implementation_filename_extensions = [".c", ".cc", ".cpp", ".tcc"]