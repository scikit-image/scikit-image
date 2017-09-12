import numpy as np

from .. import draw
from .._shared.utils import warn

import collections

Label = collections.namedtuple('Label', 'category, x1, x2, y1, y2')
Point = collections.namedtuple('Point', 'row, column')
ShapeProperties = collections.namedtuple('ShapeProperties',
                                         'min_size, max_size, color')
ImageShape = collections.namedtuple('ImageShape', 'nrows, ncols, depth')


def _generate_rectangle_mask(point, image, shape, random):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    point : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    available_width = min(image.ncols - point.column, shape.max_size)
    if available_width < shape.min_size:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image.nrows - point.row, shape.max_size)
    if available_height < shape.min_size:
        raise ArithmeticError('cannot fit shape to image')
    # Pick random widths and heights.
    r, c = random.randint(shape.min_size, available_width + 1, size=2)
    mask = np.zeros((image.nrows, image.ncols, image.depth), dtype=np.uint8)
    mask[point.row:point.row + r, point.column:point.column + c] = shape.color
    label = Label('rectangle', point.column, point.column + c, point.row,
                  point.row + r)

    return mask, label


def _generate_circle_mask(point, image, shape, random):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    point : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    if shape.min_size == 1 or shape.max_size == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape.min_size / 2.0
    max_radius = shape.max_size / 2.0
    left = point.column
    right = image.ncols - point.column
    top = point.row
    bottom = image.nrows - point.row
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    radius = random.randint(min_radius, available_radius + 1)
    mask = np.zeros((image.nrows, image.ncols, image.depth), dtype=np.uint8)
    circle = draw.circle(point.row, point.column, radius)
    mask[circle] = shape.color
    label = Label('circle', point.column - radius + 1, point.column + radius,
                  point.row - radius + 1, point.row + radius)

    return mask, label


def _generate_triangle_mask(point, image, shape, random):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    point : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    mask : 3-D array
        The shape mask that can be applied to an image.
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    """
    if shape.min_size == 1 or shape.max_size == 1:
        raise ValueError('dimension must be > 1 for circles')
    available_side = min(image.ncols - point.column, point.row + 1,
                         shape.max_size)
    if available_side < shape.min_size:
        raise ArithmeticError('cannot fit shape to image')
    side = random.randint(shape.min_size, available_side + 1)
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    mask = np.zeros((image.nrows, image.ncols, image.depth), dtype=np.uint8)
    triangle = draw.polygon([
        point.row,
        point.row - triangle_height,
        point.row,
    ], [
        point.column,
        point.column + side // 2,
        point.column + side,
    ])
    mask[triangle] = shape.color
    label = Label('triangle', point.column, point.column + side,
                  point.row - triangle_height, point.row)

    return mask, label


# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


def _generate_random_color(gray, min_pixel_intensity, random):
    """Generates a random color array.

    Parameters
    ----------
    gray : bool
        If `True`, the color will be a scalar, else a 3-element array.
    min_pixel_intensity : [0-255] int
        The lower bound for the pixel values.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ValueError
        When the min_pixel_intensity is not in the interval [0, 255].

    Returns
    -------
    color : scalar or array
        If gray is True, a single random scalar in the range of
        [min_pixel_intensity, 255], else an array of three elements, each in
        the range of [min_pixel_intensity, 255].

    """
    if not (0 <= min_pixel_intensity <= 255):
        raise ValueError('Minimum intensity must be in interval [0, 255]')
    if gray:
        return random.randint(min_pixel_intensity, 255)
    return random.randint(min_pixel_intensity, 255, size=3)


def generate_shapes(image_shape,
                    max_shapes,
                    min_shapes=1,
                    min_size=2,
                    max_size=None,
                    gray=False,
                    shape=None,
                    min_pixel_intensity=0,
                    allow_overlap=False,
                    number_of_attemps_to_fit_shape=100,
                    random_seed=None):
    """Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.

    Parameters
    ----------
    image_shape : (int, int)
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int
        The minimum dimension of each shape to fit into the image.
    max_size : int
        The maximum dimension of each shape to fit into the image.
    gray : bool
        If `True`, generate 1-D monochrome images, else 3-D RGB images.
    shape : {rectangle, circle, triangle, None} str
        The name of the shape to generate or `None` to pick random ones.
    min_pixel_intensity : [0-255] int
        The minimum pixel value for colors.
    allow_overlap : bool
        If `True`, allow shapes to overlap.
    number_of_attemps_to_fit_shape : int
        How often to attempt to fit a shape into the image before skipping it.
    seed : int
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of Label namedtuples, one per shape in the image.

    Examples
    --------
    >>> import skimage.data
    >>> image, labels = skimage.data.generate_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [Label(category='circle', x1=22, x2=25, y1=18, y2=21),
     Label(category='triangle', x1=5, x2=13, y1=6, y2=13)]
    """
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    random = np.random.RandomState(random_seed)
    user_shape = shape
    image_shape = ImageShape(
        nrows=image_shape[0], ncols=image_shape[1], depth=1 if gray else 3)
    image = np.ones(image_shape, dtype=np.uint8) * 255
    labels = []
    for _ in range(random.randint(min_shapes, max_shapes + 1)):
        color = _generate_random_color(gray, min_pixel_intensity, random)
        if user_shape is None:
            shape_generator = random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        shape = ShapeProperties(min_size, max_size, color)
        for _ in range(number_of_attemps_to_fit_shape):
            # Pick start coordinates.
            column = random.randint(image_shape.ncols)
            row = random.randint(image_shape.nrows)
            point = Point(row, column)
            try:
                mask, label = shape_generator(point, image_shape, shape,
                                              random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            # If image[mask.nonzero()].min() == 255 we haven't touched the mask
            # in the coordinates where the new mask is non-zero.
            if allow_overlap or image[mask.nonzero()].min() == 255:
                image[mask.nonzero()] = 255
                image = (image - mask).clip(0, 255)
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    return image, labels
