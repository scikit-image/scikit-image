"""Pytest configuration for doctests in the documentation pages.

``make doctest-pages`` runs the ``>>>`` blocks in the pages listed in the
Makefile's ``DOCTEST_PAGES``.  Collection is textual, so it also picks up
blocks inside ``.. plot::`` directives.  Sphinx gives those blocks ``np`` and
``plt`` through the plot directive's ``plot_pre_code``; pytest does not, so
supply the same two names here.
"""

import pytest


@pytest.fixture(autouse=True)
def _plot_directive_namespace(doctest_namespace):
    """Add the names ``matplotlib.sphinxext.plot_directive`` injects."""
    import matplotlib
    import numpy as np

    # Non-interactive: these doctests run headless under pytest.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doctest_namespace["np"] = np
    doctest_namespace["plt"] = plt
