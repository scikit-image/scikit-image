from _skimage2.draw.draw import (
    bezier_curve as bezier_curve,
    circle_perimeter as circle_perimeter,
    circle_perimeter_aa as circle_perimeter_aa,
    disk as disk,
    ellipse as ellipse,
    ellipse_perimeter as ellipse_perimeter,
    line as line,
    line_aa as line_aa,
    polygon as polygon,
    polygon_perimeter as polygon_perimeter,
    rectangle as rectangle,
    rectangle_perimeter as rectangle_perimeter,
    set_color as set_color,
)  # noqa: F401

__all__ = [
    'bezier_curve',
    'circle_perimeter',
    'circle_perimeter_aa',
    'disk',
    'ellipse',
    'ellipse_perimeter',
    'line',
    'line_aa',
    'polygon',
    'polygon_perimeter',
    'rectangle',
    'rectangle_perimeter',
    'set_color',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
