import pytest
import numpy as np
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_ensure_spacing_trivial(p):
    # --- Empty input
    assert ensure_spacing([], p_norm=p) == []

    # --- Verified spacing
    coord = np.random.randn(100, 2)

    # --- 0 spacing
    assert np.array_equal(coord, ensure_spacing(coord, spacing=0, p_norm=p))

    # Spacing is chosen to be half the minimum distance
    spacing = pdist(coord, minkowski, p=p).min() * 0.5

    out = ensure_spacing(coord, spacing=spacing, p_norm=p)

    assert np.array_equal(coord, out)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_ensure_spacing_nD(ndim):
    coord = np.ones((5, ndim))

    expected = np.ones((1, ndim))

    assert np.array_equal(ensure_spacing(coord), expected)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_ensure_spacing_p_norm(p):
    coord = np.random.randn(100, 2)

    # --- Consider the average distance btween the point as spacing
    spacing = np.median(pdist(coord, minkowski, p))
    out = ensure_spacing(coord, spacing=spacing, p_norm=p)

    assert pdist(out, minkowski, p).min() > spacing
