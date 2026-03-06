"""
This conftest is required to set the numpy print options
to legacy mode for doctests.
"""

import pytest


@pytest.fixture(autouse=True)
def handle_np2():
    # TODO: remove once we require numpy >= 2
    #       Just keep in mind that we'll have to update the docstrings
    #       everywhere once we do. E.g., `7` becomes `np.int64(7)`, and
    #       `True` becomes `np.True_`.
    try:
        import numpy as np

        np.set_printoptions(legacy="1.21")
    except ImportError:
        pass
