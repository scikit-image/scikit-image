"""Utilities for generating coordinates of different shapes and curves.

Use this module for generating coordinates of a bezier curve, circle, ellipse, rectangle, 
generating ellipsoid, calculating surface-area and volume of ellipsoid, 
drawing lines, setting pixel color at a coordinate.

...

Using this module following functionalities can be accessed


bezier_curve(r0, c0, r1, c1, ...)     Generate Bezier curve coordinates.

circle_perimeter(r, c, radius)        Generate circle perimeter coordinates.

circle_perimeter_aa(r, c, radius)     Generate anti-aliased circle perimeter coordinates.

disk(center, radius, *[, shape])      Generate coordinates of pixels within circle.

ellipse(r, c, r_radius, c_radius)     Generate coordinates of pixels within ellipse.

ellipse_perimeter(r, c, ...[, ...])   Generate ellipse perimeter coordinates.

ellipsoid(a, b, c[, spacing, ...])    Generates ellipsoid with semimajor axes aligned with grid dimensions on grid with specified spacing.

ellipsoid_stats(a, b, c)              Calculates analytical surface area and volume for ellipsoid with semimajor axes aligned with grid dimensions of specified spacing.

line(r0, c0, r1, c1)                  Generate line pixel coordinates.

line_aa(r0, c0, r1, c1)               Generate anti-aliased line pixel coordinates.

line_nd(start, stop, *[, ...])        Draw a single-pixel thick line in n dimensions.

polygon(r, c[, shape])                Generate coordinates of pixels within polygon.

polygon2mask(image_shape, polygon)    Compute a mask from polygon.

polygon_perimeter(r, c[, ...])        Generate polygon perimeter coordinates.

random_shapes(image_shape, ...)       Generate an image with random shapes, labeled with bounding boxes.

rectangle(start[, end, extent, ...])  Generate coordinates of pixels within a rectangle.

rectangle_perimeter(start[, ...])     Generate coordinates of pixels that are exactly around a rectangle.

set_color(image, coords, color)       Set pixel color in the image at the given coordinates.

"""
from .draw import (ellipse, set_color, polygon_perimeter,
                   line, line_aa, polygon, ellipse_perimeter,
                   circle_perimeter, circle_perimeter_aa,
                   disk, bezier_curve, rectangle, rectangle_perimeter)
from .draw3d import ellipsoid, ellipsoid_stats
from ._draw import _bezier_segment
from ._random_shapes import random_shapes
from ._polygon2mask import polygon2mask

from .draw_nd import line_nd

__all__ = ['line',
           'line_aa',
           'line_nd',
           'bezier_curve',
           'polygon',
           'polygon_perimeter',
           'ellipse',
           'ellipse_perimeter',
           'ellipsoid',
           'ellipsoid_stats',
           'circle_perimeter',
           'circle_perimeter_aa',
           'disk',
           'set_color',
           'random_shapes',
           'rectangle',
           'rectangle_perimeter',
           'polygon2mask']
