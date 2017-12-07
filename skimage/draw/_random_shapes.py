import numpy as np

from . import polygon as draw_polygon, circle as draw_circle
from .._shared.utils import warn


def _generate_rectangle_mask(point, image, shape, random):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    available_width = min(image[1] - point[1], shape[1])
    if available_width < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], shape[1])
    if available_height < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    # Pick random widths and heights.
    r = random.randint(shape[0], available_height + 1)
    c = random.randint(shape[0], available_width + 1)
    rectangle = draw_polygon([
        point[0],
        point[0] + r,
        point[0] + r,
        point[0],
    ], [
        point[1],
        point[1],
        point[1] + c,
        point[1] + c,
    ])
    label = ('rectangle', ((point[0], point[0] + r), (point[1], point[1] + c)))

    return rectangle, label


def _generate_circle_mask(point, image, shape, random):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    radius = random.randint(min_radius, available_radius + 1)
    circle = draw_circle(point[0], point[1], radius)
    label = ('circle', ((point[0] - radius + 1, point[0] + radius),
                        (point[1] - radius + 1, point[1] + radius)))

    return circle, label


def _generate_triangle_mask(point, image, shape, random):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of a down-pointing triangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0] + 1, shape[1])
    if available_side < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    side = random.randint(shape[0], available_side + 1)
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    triangle = draw_polygon([
        point[0],
        point[0] - triangle_height,
        point[0],
    ], [
        point[1],
        point[1] + side // 2,
        point[1] + side,
    ])
    label = ('triangle', ((point[0] - triangle_height, point[0]),
                          (point[1], point[1] + side)))

    return triangle, label


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
        return random.randint(min_pixel_intensity, 256)
    return random.randint(min_pixel_intensity, 256, size=3)


def random_shapes(image_shape,
                  max_shapes,
                  min_shapes=1,
                  min_size=2,
                  max_size=None,
                  gray=False,
                  shape=None,
                  min_pixel_intensity=0,
                  allow_overlap=False,
                  num_trials=100,
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
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    gray : bool, optional
        If `True`, generate monochrome images, else color images with multiple
        channels.
    shape : {rectangle, circle, triangle, None} str, optional
        The name of the shape to generate or `None` to pick random ones.
    min_pixel_intensity : [0-255] int, optional
        The minimum pixel value for colors.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    seed : int, optional
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a (category,
        ((r0, r1), (c0, c1))) tuple specifying the category and bounding box
        coordinates of the shape.

    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
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
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    """
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    random = np.random.RandomState(random_seed)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], 1 if gray else 3)
    image = np.ones(image_shape, dtype=np.uint8) * 255
    filled = np.zeros(image_shape, dtype=bool)
    labels = []
    for _ in range(random.randint(min_shapes, max_shapes + 1)):
        color = _generate_random_color(gray, min_pixel_intensity, random)
        if user_shape is None:
            shape_generator = random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        shape = (min_size, max_size)
        for _ in range(num_trials):
            # Pick start coordinates.
            column = random.randint(image_shape[1])
            row = random.randint(image_shape[0])
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape,
                                                 random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = color
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    return image, labels
