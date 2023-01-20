# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys
from warnings import filterwarnings

import plotly.io as pio
import skimage
from packaging.version import parse
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.utils import _has_optipng

filterwarnings(
    "ignore", message="Matplotlib is currently using agg", category=UserWarning
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skimage"
copyright = "2013, the scikit-image team"

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
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
]

autosummary_generate = True
templates_path = ["_templates"]
source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

exclude_trees = []
default_role = "autolink"
pygments_style = "sphinx"

# -- Sphinx-gallery configuration --------------------------------------------

v = parse(release)
if v.release is None:
    raise ValueError(f"Ill-formed version: {version!r}. Version should follow PEP440")

if v.is_devrelease:
    binder_branch = "main"
else:
    major, minor = v.release[:2]
    binder_branch = f"v{major}.{minor}.x"

# set plotly renderer to capture _repr_html_ for sphinx-gallery

pio.renderers.default = "sphinx_gallery_png"

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
    "binder": {
        # Required keys
        "org": "scikit-image",
        "repo": "scikit-image",
        "branch": binder_branch,  # Can be any branch, tag, or commit hash
        "binderhub_url": "https://mybinder.org",  # Any URL of a binderhub.
        "dependencies": ["../../.binder/requirements.txt", "../../.binder/runtime.txt"],
        # Optional keys
        "use_jupyter_lab": False,
    },
    # Remove sphinx_gallery_thumbnail_number from generated files
    "remove_config_comments": True,
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

html_theme = "scikit-image"
html_theme_path = ["themes"]
html_title = f"skimage v{version} docs"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": ["searchbox.html", "navigation.html", "localtoc.html", "versions.html"],
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
_python_version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
_python_doc_base = "https://docs.python.org/" + _python_version_str
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "./_intersphinx/numpy-objects.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/",
        (None, "./_intersphinx/scipy-objects.inv"),
    ),
    "sklearn": (
        "https://scikit-learn.org/stable/",
        (None, "./_intersphinx/sklearn-objects.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        (None, "./_intersphinx/matplotlib-objects.inv"),
    ),
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
        except:
            return None

    # Strip decorators which would resolve to the source of the decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, start_line = inspect.getsourcelines(obj)
    except:
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
