"""
This conftest is required to set the numpy print options
to legacy mode for doctests
"""

import pytest


@pytest.fixture(autouse=True)
def handle_np2():
    # TODO: remove when we require numpy >= 2
    try:
        import numpy as np

        np.set_printoptions(legacy="1.21")
    except ImportError:
        pass
