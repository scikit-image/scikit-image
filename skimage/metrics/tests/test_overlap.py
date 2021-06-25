from skimage.metrics.overlap import (
        BoundingBox,
        disjoint,
        intersect,
        intersection_over_union,
        )
from skimage._shared import testing

height1, width1 = 2, 4
r1 = BoundingBox((0, 0), shape=(height1, width1))

height2 = width2 = 3
r2 = BoundingBox((2, 3), shape=(height2, width2))  # share part of top side with r1

r3 = BoundingBox((0, 0), bottom_right=(2, 3))  # included in r1, intersect with r2 in (2,3)
r4 = BoundingBox((2, 3), shape=(0, 0))    # 0-area rectangle
r5 = BoundingBox((5, 5), shape=(5, 5))

r3D = BoundingBox((1,2,3), shape=(2,2,2))

rect_inter13 = intersect(r1, r3)


def test_area():
    assert r1.area == height1 * width1
    assert r2.area == height2 * width2
    assert r3.area == 2 * 3
    with testing.raises(NotImplementedError):
        r3D.area

def test_volume():
    assert r3D.volume == 2**3
    with testing.raises(NotImplementedError):
        r1.volume

def test_constructor_dimensions():
    assert tuple(r1.top_left) == (0, 0)
    assert tuple(r1.bottom_right) == (height1, width1)
    with testing.raises(ValueError):
        BoundingBox((0, 0), shape=(-5, 5))


def test_constructor_bottom_corner():
    assert tuple(r3.top_left) == (0, 0)
    assert tuple(r3.shape) == (2, 3)
    with testing.raises(ValueError):
        BoundingBox((2, 2), bottom_right=(0, 0))

def test_disjoint():
    assert disjoint(r1, r5)

def test_intersection():
    assert intersect(r1, r1) == r1      # self-intersection is self
    assert intersect(r4, r4) == r4      # also for 0-area rectangles
    assert intersect(r1, r2).area == 0  # edge intersection
    assert intersect(r2, r3).area == 0  # corner intersection
    assert intersect(r2, r4) == r4      # corner intersection with 0-area rectangle
    assert intersect(r1, r4) == r4      # 0-area rectangle included within another rectangle
    assert intersect(r1, r5) is None


def test_eq_operator():  # Intersection rectangle and == comparison
    assert rect_inter13 == r3
    assert r1 != r3D  # BoundingBoxes of different shape

def test_eq_other_obj():
    with testing.raises(TypeError):
        _ = r2 == (1, 2, 3, 4)


def test_str():
    assert str(r3) == 'BoundingBox((0, 0), bottom_right=(2, 3))'


def test_IoU():
    # IoU for BBox with themselves
    intersection_over_union(r1, r1) == 1
    intersection_over_union(r4, r4) == 1  # also for 0-area rectangles
    intersection_over_union(r1, r5) == 0

    union_area13 = r1.area + r3.area - rect_inter13.area
    assert intersection_over_union(r1, r3) == rect_inter13.area / union_area13
