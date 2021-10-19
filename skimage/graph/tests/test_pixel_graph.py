import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel

mask = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=bool)
image = np.random.default_rng().random(mask.shape)


def test_small_graph():
    g, n = pixel_graph(mask, connectivity=2)
    assert g.shape == (4, 4)
    assert len(g.data) == 8
    np.testing.assert_allclose(np.unique(g.data), [1, np.sqrt(2)])
    np.testing.assert_array_equal(n, [0, 4, 5, 7])


def test_central_pixel():
    g, n = pixel_graph(mask, connectivity=2)
    px, ds = central_pixel(g, n, shape=mask.shape)
    np.testing.assert_array_equal(px, (1, 1))
    s2 = np.sqrt(2)
    np.testing.assert_allclose(ds, [s2*3 + 2, s2 + 2, s2*2 + 2, s2*2 + 2])
    px, _ = central_pixel(g, n)
    assert px == 4


def test_edge_function():
    def edge_func(values_src, values_dst, distances):
        return np.abs(values_src - values_dst) + distances

    g, n = pixel_graph(
            image, mask=mask, connectivity=2, edge_function=edge_func
            )
    s2 = np.sqrt(2)
    np.testing.assert_allclose(g[0, 1], np.abs(image[0, 0] - image[1, 1]) + s2)
    np.testing.assert_allclose(g[1, 2], np.abs(image[1, 1] - image[1, 2]) + 1)
    np.testing.assert_array_equal(n, [0, 4, 5, 7])


if __name__ == '__main__':
    test_edge_function()
