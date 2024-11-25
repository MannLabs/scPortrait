# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/scportrait"))


def setup(app):
    app.add_css_file("_static/hide_links.css")


# -- Project information -----------------------------------------------------

project = "scPortrait"
copyright = "2024 Sophia Mädler and Niklas Schmacke"
author = "Sophia Mädler and Niklas Schmacke"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinx_rtd_theme",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]
exclude_patterns = ["**.ipynb_checkpoints"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # ignore automatically generated files by sphinx-gallery
    "auto_examples/**.ipynb",
    "auto_examples/**.json",
    "auto_examples/**.py",
    "auto_examples/**.md5",
]

# autodoc_mock_imports = []
autodoc_mock_imports = []  # type: ignore

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "collapse_navigation": False,
    # "sticky_navigation": True,
    "navigation_depth": 4,
    # "logo_only": True,
    "logo": {
        "image_light": "_static/scPortrait_logo_light.svg",
        "image_dark": "_static/scPortrait_logo_dark.svg",
    },
}

html_title = "scPortrait"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_default_options = {
    "member-order": "bysource",
}

html_favicon = "favicon.png"
html_logo = "_static/scPortrait_logo_light.svg"

# set up sphinx gallery
sg_examples_dir = [
    "../examples/code_snippets",
]
sg_gallery_dir = [
    "auto_examples/code_snippets",
]

sphinx_gallery_conf = {
    "doc_module": "scportrait",
    "reference_url": {"scportrait": None},
    "show_memory": False,
    "examples_dirs": sg_examples_dir,
    "gallery_dirs": sg_gallery_dir,
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "write_computation_times": False,
    "min_reported_time": 240,
    "reset_modules": ("matplotlib"),
    "download_all_examples": False,
    "promote_jupyter_magic": True,
}

#turn off execution of notebooks during build of docs
nbsphinx_execute = 'never'