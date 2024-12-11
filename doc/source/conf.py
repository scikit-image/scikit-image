# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import inspect
import os
import sys
from warnings import filterwarnings

import plotly.io as pio
import skimage
from intersphinx_registry import get_intersphinx_mapping
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
copyright = f"2013-{date.today().year}, the scikit-image team"

with open("../../skimage/__init__.py") as f:
    setup_lines = f.readlines()
version = "vUndefined"
for l in setup_lines:
    if l.startswith("__version__ ="):
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
    "pytest_doctestplus.sphinx.doctestplus",
    "skimage_extensions",
]

autosummary_generate = True
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}

show_warning_types = True
suppress_warnings = [
    # Ignore new warning in Sphinx 7.3.0 while pickling environment:
    #   WARNING: cannot cache unpickable configuration value: 'sphinx_gallery_conf'
    "config.cache",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

exclude_trees = []
default_role = "py:obj"
pygments_style = "sphinx"

# -- Sphinx-gallery configuration --------------------------------------------

v = parse(release)
if v.release is None:
    raise ValueError(f"Ill-formed version: {version!r}. Version should follow PEP440")

if v.is_devrelease:
    binder_branch = "main"
else:
    binder_branch = f"v{release}"

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
    # `True` defaults to the number of jobs used by Sphinx (see its flag `-j`)
    "parallel": True,
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
        "text": "scikit-image",
        "link": "https://scikit-image.org",
    },
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-image/scikit-image",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scikit-image/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "show_prev_next": False,
    "switcher": {
        "json_url": ("https://scikit-image.org/docs/dev/_static/version_switcher.json"),
        "version_match": "dev" if "dev" in version else version,
    },
    "show_version_warning_banner": True,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # Other
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
    "analytics": {
        "plausible_analytics_domain": "scikit-image.org",
        "plausible_analytics_url": ("https://views.scientific-python.org/js/script.js"),
    },
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
latex_elements["preamble"] = r"""
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
# ...
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "python",
        "numpy",
        "neps",
        "scipy",
        "sklearn",
        "matplotlib",
        "networkx",
        "plotly",
        "seaborn",
    }
)

# Do not (yet) use nitpicky mode for checking cross-references
nitpicky = False
# nitpick_ignore is only considered when nitpicky=True
nitpick_ignore = [
    (
        "py:class",
        "skimage.transform._geometric._GeometricTransform",
    ),  # skimage.transform._geometric.{FundamentalMatrixTransform,PiecewiseAffineTransform,PolynomialTransform,ProjectiveTransform}
    (
        "py:class",
        "skimage.feature.util.DescriptorExtractor",
    ),  # skimage.feature.{censure.CENSURE/orb.ORB/sift.SIFT}
    (
        "py:class",
        "skimage.feature.util.FeatureDetector",
    ),  # skimage.feature.{censure.CENSURE/orb.ORB/sift.SIFT}
    (
        "py:class",
        "skimage.measure.fit.BaseModel",
    ),  # skimage.measure.fit.{CircleModel/EllipseModel/LineModelND}
    ("py:exc", "NetworkXError"),  # networkx.classes.graph.Graph.nbunch_iter
    ("py:obj", "Graph"),  # networkx.classes.graph.Graph.to_undirected
    ("py:obj", "Graph.__iter__"),  # networkx.classes.graph.Graph.nbunch_iter
    ("py:obj", "__len__"),  # networkx.classes.graph.Graph.{number_of_nodes/order}
    (
        "py:class",
        "_GeometricTransform",
    ),  # skimage.transform._geometric.estimate_transform
    ("py:obj", "convert"),  # skimage.graph._rag.RAG.__init__
    ("py:obj", "skimage.io.collection"),  # (generated) doc/source/api/skimage.io.rst
    (
        "py:obj",
        "skimage.io.manage_plugins",
    ),  # (generated) doc/source/api/skimage.io.rst
    ("py:obj", "skimage.io.sift"),  # (generated) doc/source/api/skimage.io.rst
    ("py:obj", "skimage.io.util"),  # (generated) doc/source/api/skimage.io.rst
]
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
