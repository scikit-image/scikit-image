from skimage.draw import line_nd


def test_empty_line():
    coords = line_nd((1, 1, 1), (1, 1, 1))
    assert len(coords) == 3
    assert all(len(c) == 0 for c in coords)
