from skimage.metrics.overlap import (
        Rectangle,
        intersect,
        intersection_over_union,
        )
from skimage._shared import testing

height1, width1 = 2, 4
rectangle1 = Rectangle((0, 0), dimensions=(height1, width1))

height2 = width2 = 3
rectangle2 = Rectangle((2, 3), dimensions=(height2, width2))

rectangle3 = Rectangle((0, 0), bottom_right=(2, 2))

rectangle4 = Rectangle((10, 10), dimensions=(5, 5))

rect_inter13 = intersect(rectangle1, rectangle3)


def test_area():
    assert rectangle1.area == height1 * width1
    assert rectangle2.area == height2 * width2
    assert rectangle3.area == 2 * 2


def test_constructor_dimensions():
    assert tuple(rectangle1.top_left) == (0, 0)
    assert tuple(rectangle1.bottom_right) == (height1, width1)


def test_constructor_bottom_corner():
    assert tuple(rectangle3.top_left) == (0, 0)
    assert tuple(rectangle3.dimensions) == (2, 2)


def test_intersection():
    assert not intersect(rectangle1, rectangle2)
    assert intersect(rectangle1, rectangle3)


def test_eq_operator():  # Intersection rectangle and == comparison
    assert rect_inter13 == Rectangle((0, 0), bottom_right=(2, 2))


def test_eq_other_obj():
    with testing.raises(TypeError):
        _ = rectangle2 == (1, 2, 3, 4)


def test_str():
    assert str(rectangle3) == 'Rectangle((0, 0), bottom_right=(2, 2))'


def test_IoU():
    union_area13 = rectangle1.area + rectangle3.area - rect_inter13.area
    assert intersection_over_union(rectangle1, rectangle3) == rect_inter13.area / union_area13
