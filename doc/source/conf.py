# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import inspect
import os
import sys
from warnings import filterwarnings
import warnings

import plotly.io as pio
import skimage
from packaging.version import parse
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.utils import _has_optipng
from sphinx_gallery.notebook import add_markdown_cell, add_code_cell

filterwarnings(
    "ignore", message="Matplotlib is currently using agg", category=UserWarning
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skimage"
copyright = f"2013-{date.today().year}, the scikit-image team"

with open("../../skimage/__init__.py") as f:
    setup_lines = f.readlines()
version = "vUndefined"
for l in setup_lines:
    if l.startswith("__version__"):
        version = l.split("'")[1]
        break

release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

curpath = os.path.dirname(__file__)
sys.path.append(os.path.join(curpath, "..", "ext"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "doi_role",
    "numpydoc",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "skimage_extensions",
]

autosummary_generate = True
templates_path = ["_templates"]
source_suffix = ".rst"

try:
    import jupyterlite_sphinx  # noqa: F401

    extensions.append("jupyterlite_sphinx")
except ImportError:
    # In some cases we don't want to require jupyterlite_sphinx to be installed,
    # e.g. the doc-min-dependencies build
    warnings.warn(
        "jupyterlite_sphinx is not installed, you need to install it "
        "if you want JupyterLite links to appear in each example"
    )

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

exclude_trees = []
default_role = "autolink"
pygments_style = "sphinx"

# -- Sphinx-gallery configuration --------------------------------------------

v = parse(release)
if v.release is None:
    raise ValueError(f"Ill-formed version: {version!r}. Version should follow PEP440")

# set plotly renderer to capture _repr_html_ for sphinx-gallery
pio.renderers.default = "sphinx_gallery_png"


def notebook_modification_function(notebook_content, notebook_filename):
    notebook_content_str = str(notebook_content)
    warning_template = "\n".join(
        [
            "<div class='alert alert-{message_class}'>",
            "",
            "# JupyterLite warning",
            "",
            "{message}",
            "</div>",
        ]
    )

    message_class = "warning"
    message = (
        "Running the scikit-image examples in JupyterLite is experimental and you may"
        " encounter some unexpected behavior.\n\nThe main difference is that imports"
        " will take a lot longer than usual, for example the first `import skimage` can"
        " take roughly 10-20s.\n\nIf you notice problems, feel free to open an"
        " [issue](https://github.com/scikit-image/scikit-image/issues/new/choose)"
        " about it."
    )

    markdown = warning_template.format(message_class=message_class, message=message)

    dummy_notebook_content = {"cells": []}
    add_markdown_cell(dummy_notebook_content, markdown)

    code_lines = []

    if "seaborn" in notebook_content_str:
        code_lines.append("%pip install seaborn")
    if "plotly" in notebook_content_str:
        code_lines.append("%pip install plotly")
    if "data." in notebook_content_str or ".data" in notebook_content_str:
        code_lines.extend(
            [
                # lzma needs to be imported so that %pip install pooch works
                "import lzma",
                # pooch depends on requests and need to be installed before
                # pyodide_http.patch_all() is called
                "%pip install pooch",
                "import pooch",
                "%pip install pyodide-http",
                "import pyodide_http",
                "pyodide_http.patch_all()",
            ]
        )
    # Use cdn.statically.io for CORS proxy that supports gitlab.com
        code_lines.extend(
    r"""
import re

import skimage.data._registry

new_registry_urls = {
    k: re.sub(
        r'https://gitlab.com/(.+)/-/raw(.+)',
        r'https://cdn.statically.io/gl/\1\2',
        url
    )
    for k, url in skimage.data._registry.registry_urls.items()
}
skimage.data._registry.registry_urls = new_registry_urls
    """.splitlines()
        )

    if code_lines:
        code_lines = ["# JupyterLite-specific code"] + code_lines
        code = "\n".join(code_lines)
        add_code_cell(dummy_notebook_content, code)

    notebook_content["cells"] = (
        dummy_notebook_content["cells"] + notebook_content["cells"]
    )


sphinx_gallery_conf = {
    "doc_module": ("skimage",),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "api",
    "reference_url": {"skimage": None},
    "image_scrapers": ("matplotlib", plotly_sg_scraper),
    "subsection_order": ExplicitOrder(
        [
            "../examples/data",
            "../examples/numpy_operations",
            "../examples/color_exposure",
            "../examples/edges",
            "../examples/transform",
            "../examples/registration",
            "../examples/filters",
            "../examples/features_detection",
            "../examples/segmentation",
            "../examples/applications",
            "../examples/developers",
        ]
    ),
    # Remove sphinx_gallery_thumbnail_number from generated files
    "remove_config_comments": True,
    "jupyterlite": {"notebook_modification_function": notebook_modification_function},
    # Can be disabled during development to accelerate build
    "plot_gallery": True
}


if _has_optipng():
    # This option requires optipng to compress images
    # Optimization level between 0-7
    # sphinx-gallery default: -o7
    # optipng default: -o2
    # We choose -o1 as it produces a sufficient optimization
    # See #4800
    sphinx_gallery_conf["compress_images"] = ("images", "thumbnails", "-o1")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/branding.html

html_theme = "pydata_sphinx_theme"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_css_files = ['theme_overrides.css']

html_theme_options = {
    # Navigation bar
    "logo": {
        "alt_text": (
            "scikit-image's logo, showing a snake's head overlayed with green "
            "and orange"
        ),
        "text": "scikit-image"
    },
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scikit-image/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "show_prev_next": False,
    "switcher": {
        "json_url": "https://scikit-image.org/docs/dev/_static/version_switcher.json",
        "version_match": "dev" if "dev" in version else version,
    },
    "github_url": "https://github.com/scikit-image/scikit-image",
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # Other
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark",
}

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    # "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
    "auto_examples/index": []  # Hide sidebar in example gallery
}

# Output file base name for HTML help builder.
htmlhelp_basename = "scikitimagedoc"


# -- Options for LaTeX output --------------------------------------------------

latex_font_size = "10pt"
latex_documents = [
    (
        "index",
        "scikit-image.tex",
        "The scikit-image Documentation",
        "scikit-image development team",
        "manual",
    ),
]
latex_elements = {}
latex_elements[
    "preamble"
] = r"""
\usepackage{enumitem}
\setlistdepth{100}

\usepackage{amsmath}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}

% In the parameters section, place a newline after the Parameters header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

"""
latex_domain_indices = False

# -- numpydoc extension -------------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "neps": ("https://numpy.org/neps/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Source code links -------------------------------------------------------

# Function courtesy of NumPy to return URLs containing line numbers
def linkcode_resolve(domain, info):

    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Strip decorators which would resolve to the source of the decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None

    if not fn:
        return None

    try:
        source, start_line = inspect.getsourcelines(obj)
    except OSError:
        linespec = ""
    else:
        stop_line = start_line + len(source) - 1
        linespec = f"#L{start_line}-L{stop_line}"

    fn = os.path.relpath(fn, start=os.path.dirname(skimage.__file__))

    if "dev" in skimage.__version__:
        return (
            "https://github.com/scikit-image/scikit-image/blob/"
            f"main/skimage/{fn}{linespec}"
        )
    else:
        return (
            "https://github.com/scikit-image/scikit-image/blob/"
            f"v{skimage.__version__}/skimage/{fn}{linespec}"
        )


# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    # Enable fieldlist to allow for Field Lists like in rST (e.g., :orphan:)
    "fieldlist",
]
