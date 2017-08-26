import numpy as np

from .. import draw
from .._shared.utils import warn

import collections

Label = collections.namedtuple('Label', 'category, x1, x2, y1, y2')


def _generate_rectangle_mask(x_0,
                             y_0,
                             image_width,
                             image_height,
                             image_depth,
                             color,
                             min_dimension,
                             max_dimension):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    x_0 : int
        The x coordinate (column) of the top left corner of the rectangle.
    y_0 : int
        The y coordinate (row) of the top left corner of the rectangle.
    image_width: int
        The width of the image into which this shape will be fit.
    image_height: int
        The height of the image into which this shape will be fit.
    image_depth: int
        The depth of the image into which this shape will be fit.
    color: array
        The pixel array to assign to the shape.
    min_dimension:
        The minimum dimension expected for the shape.
    max_dimension:
        The maximum dimension expected for the shape.

    Raises
    -----
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label: Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    available_width = min(image_width - x_0, max_dimension)
    if available_width < min_dimension:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image_height - y_0, max_dimension)
    if available_height < min_dimension:
        raise ArithmeticError('cannot fit shape to image')
    # Pick random widths and heights.
    w = np.random.randint(min_dimension, available_width + 1)
    h = np.random.randint(min_dimension, available_height + 1)
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    mask[y_0:y_0 + h, x_0:x_0 + w] = color
    assert mask.sum() > 0
    label = Label('rectangle', x_0, x_0 + w, y_0, y_0 + h)

    return mask, label


def _generate_circle_mask(x_0,
                          y_0,
                          image_width,
                          image_height,
                          image_depth,
                          color,
                          min_dimension,
                          max_dimension):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    x_0 : int
        The x coordinate (column) of the center of the circle.
    y_0 : int
        The y coordinate (row) of the center of the circle.
    image_width: int
        The width of the image into which this shape will be fit.
    image_height: int
        The height of the image into which this shape will be fit.
    image_depth: int
        The depth of the image into which this shape will be fit.
    color: array
        The pixel array to assign to the shape.
    min_dimension:
        The minimum dimension expected for the shape.
    max_dimension:
        The maximum dimension expected for the shape.

    Raises
    -----
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label: Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    if min_dimension == 1 or max_dimension == 1:
        raise ValueError('dimension must be > 1 for circles')
    min_dimension /= 2
    max_dimension /= 2
    left = x_0
    right = image_width - x_0
    top = y_0
    bottom = image_height - y_0
    available_radius = min(left, right, top, bottom, max_dimension)
    if available_radius < min_dimension:
        raise ArithmeticError('cannot fit shape to image')
    radius = np.random.randint(min_dimension, available_radius + 1)
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    circle = draw.circle(y_0, x_0, radius)
    mask[circle] = color
    assert mask.sum() > 0
    label = Label('circle', x_0 - radius + 1, x_0 + radius, y_0 - radius + 1,
                  y_0 + radius)

    return mask, label


def _generate_triangle_mask(x_0,
                            y_0,
                            image_width,
                            image_height,
                            image_depth,
                            color,
                            min_dimension,
                            max_dimension):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    x_0 : int
        The x coordinate (column) of the bottom left corner of the triangle.
    y_0 : int
        The y coordinate (row) of the bottom left corner of the triangle.
    image_width: int
        The width of the image into which this shape will be fit.
    image_height: int
        The height of the image into which this shape will be fit.
    image_depth: int
        The depth of the image into which this shape will be fit.
    color: array
        The pixel array to assign to the shape.
    min_dimension:
        The minimum dimension expected for the shape.
    max_dimension:
        The maximum dimension expected for the shape.

    Raises
    -----
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label: Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    if min_dimension == 1 or max_dimension == 1:
        raise ValueError('dimension must be > 1 for circles')
    available_side = min(image_width - x_0, y_0 + 1, max_dimension)
    if available_side < min_dimension:
        raise ArithmeticError('cannot fit shape to image')
    side = np.random.randint(min_dimension, available_side + 1)
    triangle_height = int(np.ceil(np.sqrt(3 / 4) * side))
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    triangle = draw.polygon([y_0, y_0 - triangle_height, y_0],
                            [x_0, x_0 + side // 2, x_0 + side])
    mask[triangle] = color
    assert mask.sum() > 0
    label = Label('triangle', x_0, x_0 + side, y_0 - triangle_height, y_0)

    return mask, label


# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


def _generate_random_color(gray, min_pixel_intensity):
    """Generates a random color array.

    Parameters
    ----------
    gray : bool
        If true, the color will be a scalar, else a 3-element array.
    min_pixel_intensity: [0-255] int
        The lower bound for the pixel values.

    Raises
    ------
    ValueError
        When the min_pixel_intensity is not in the interval [0, 255].

    Returns
    -------
    color: scalar or array
        If gray is True, a single random scalar in the range of
        [min_pixel_intensity, 255], else an array of three elements, each in the
        range of [min_pixel_intensity, 255].

    """
    if not (0 <= min_pixel_intensity <= 255):
        raise ValueError('Minimum intensity must be in interval [0, 255]')
    if gray:
        return np.random.randint(min_pixel_intensity, 255)
    return np.random.randint(min_pixel_intensity, 255, size=3)


def generate_shapes(width,
                    height,
                    max_shapes,
                    min_shapes=1,
                    min_dimension=2,
                    max_dimension=None,
                    gray=False,
                    shape=None,
                    min_pixel_intensity=32,
                    allow_overlap=False,
                    number_of_attemps_to_fit_shape=100):
    """Generate an image with random shapes, labeled with bounding boxes.

    Images can be populated with random shapes with random dimensions, random
    locations, and random colors, with or without overlap.

    Shapes have random (x, y) starting coordinates and random dimensions bounded
    by min_dimension and max_dimension. It can occur that a randomly generated
    shape will not fit the image at all. In that case, the algorithm will try
    again with new starting coordinates a certain number of times. However, it
    also means that some images may be skipped altogether if no shapes fit. In
    that case, this function will return fewer images than requested.

    Parameters
    ----------
    width: int
        The width of the image to generate.
    height: int
        The height of the image to generate.
    max_shapes: int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes: int
        The minimum number of shapes to (attempt to) fit into the shape.
    min_dimension: int
        The minimum dimension of each shape to fit into the image.
    max_dimension: int
        The maximum dimension of each shape to fit into the image.
    gray: bool
        If true, generate 1-D monochrome images, else 3-D RGB images.
    shape: {rectangle, circle, triangle, None} str
        The name of the shape to generate or None to pick random ones.
    min_pixel_intensity: [0-255] int
        The minimum pixel value for colors.
    allow_overlap: bool
        If true, allow shapes to overlap.
    number_of_attemps_to_fit_shape: int
        How often to attempt to fit a shape into the image before skipping it.

    Returns
    -------
    image: uint8 array
        An image with the fitted shapes.
    labels: list
        A list of label namedtuples, one per shape in the image.

    """
    max_dimension = max_dimension or max(height, width)
    if min_dimension > width or min_dimension > height:
        raise ValueError(
            'Minimum dimension must be less than width and height')

    depth = 1 if gray else 3
    image = np.ones((height, width, depth), dtype=np.uint8) * 255
    labels = []
    for _ in range(np.random.randint(min_shapes, max_shapes + 1)):
        color = _generate_random_color(gray, min_pixel_intensity)
        if shape is None:
            shape_generator = np.random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[shape]
        for _ in range(number_of_attemps_to_fit_shape):
            # Pick start coordinates.
            x = np.random.randint(width)
            y = np.random.randint(height)
            try:
                mask, label = shape_generator(x,
                                              y,
                                              width,
                                              height,
                                              depth,
                                              color,
                                              min_dimension,
                                              max_dimension)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            assert mask.sum() > 0, mask
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or image[mask.nonzero()].min() == 255:
                image = (image - mask).clip(0, 255)
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    return image, labels
