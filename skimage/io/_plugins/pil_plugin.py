__all__ = ['imread', 'imsave']

import numpy as np
from six import string_types
from PIL import Image

from ...util import img_as_ubyte, img_as_uint


def imread(fname, dtype=None, img_num=None, **kwargs):
    """Load an image from file.

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.
    dtype : numpy dtype object or string specifier
       Specifies data type of array elements.
    img_num : int, optional
       Specifies which image to read in a file with multiple images
       (zero-indexed).
    kwargs : keyword pairs, optional
        Addition keyword arguments to pass through.

    Notes
    -----
    Files are read using the Python Imaging Libary.
    See PIL docs [1]_ for a list of supported formats.

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    """
    if isinstance(fname, string_types):
        with open(fname, 'rb') as f:
            im = Image.open(f)
            return pil_to_ndarray(im, dtype=dtype, img_num=img_num)
    else:
        im = Image.open(fname)
        return pil_to_ndarray(im, dtype=dtype, img_num=img_num)


def pil_to_ndarray(im, dtype=None, img_num=None):
    """Import a PIL Image object to an ndarray, in memory.

    Parameters
    ----------
    Refer to ``imread``.

    """
    try:
        # this will raise an IOError if the file is not readable
        im.getdata()[0]
    except IOError as e:
        site = "http://pillow.readthedocs.org/en/latest/installation.html#external-libraries"
        pillow_error_message = str(e)
        error_message = ('Could not load "%s" \n'
                         'Reason: "%s"\n'
                         'Please see documentation at: %s'
                         % (im.filename, pillow_error_message, site))
        raise ValueError(error_message)
    frames = []
    grayscale = None
    i = 0
    while 1:
        try:
            im.seek(i)
        except EOFError:
            break

        frame = im

        if img_num is not None and img_num != i:
            im.getdata()[0]
            i += 1
            continue

        if im.format == 'PNG' and im.mode == 'I' and dtype is None:
            dtype = 'uint16'

        if im.mode == 'P':
            if grayscale is None:
                grayscale = _palette_is_grayscale(im)

            if grayscale:
                frame = im.convert('L')
            else:
                if im.format == 'PNG' and 'transparency' in im.info:
                    frame = im.convert('RGBA')
                else:
                    frame = im.convert('RGB')

        elif im.mode == '1':
            frame = im.convert('L')

        elif 'A' in im.mode:
            frame = im.convert('RGBA')

        elif im.mode == 'CMYK':
            frame = im.convert('RGB')

        if im.mode.startswith('I;16'):
            shape = im.size
            dtype = '>u2' if im.mode.endswith('B') else '<u2'
            if 'S' in im.mode:
                dtype = dtype.replace('u', 'i')
            frame = np.fromstring(frame.tobytes(), dtype)
            frame.shape = shape[::-1]

        else:
            frame = np.array(frame, dtype=dtype)

        frames.append(frame)
        i += 1

        if img_num is not None:
            break

    if hasattr(im, 'fp') and im.fp:
        im.fp.close()

    if img_num is None and len(frames) > 1:
        return np.array(frames)
    elif frames:
        return frames[0]
    elif img_num:
        raise IndexError('Could not find image  #%s' % img_num)


def _palette_is_grayscale(pil_image):
    """Return True if PIL image in palette mode is grayscale.

    Parameters
    ----------
    pil_image : PIL image
        PIL Image that is in Palette mode.

    Returns
    -------
    is_grayscale : bool
        True if all colors in image palette are gray.
    """
    assert pil_image.mode == 'P'
    # get palette as an array with R, G, B columns
    palette = np.asarray(pil_image.getpalette()).reshape((256, 3))
    # Not all palette colors are used; unused colors have junk values.
    start, stop = pil_image.getextrema()
    valid_palette = palette[start:stop]
    # Image is grayscale if channel differences (R - G and G - B)
    # are all zero.
    return np.allclose(np.diff(valid_palette), 0)


def ndarray_to_pil(arr, format_str=None):
    """Export an ndarray to a PIL object.

    Parameters
    ----------
    Refer to ``imsave``.

    """
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

    elif format_str in ['png', 'PNG']:
        mode = 'I;16'
        mode_base = 'I'

        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)

        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = mode_base = 'L'

        else:
            arr = img_as_uint(arr)

    else:
        arr = img_as_ubyte(arr)
        mode = 'L'
        mode_base = 'L'

    try:
        array_buffer = arr.tobytes()
    except AttributeError:
        array_buffer = arr.tostring()  # Numpy < 1.9

    if arr.ndim == 2:
        im = Image.new(mode_base, arr.T.shape)
        try:
            im.frombytes(array_buffer, 'raw', mode)
        except AttributeError:
            im.fromstring(array_buffer, 'raw', mode)  # PIL 1.1.7
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            im = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            im = Image.fromstring(mode, image_shape, array_buffer)  # PIL 1.1.7
    return im


def imsave(fname, arr, format_str=None, **kwargs):
    """Save an image to disk.

    Parameters
    ----------
    fname : str or file-like object
        Name of destination file.
    arr : ndarray of uint8 or float
        Array (image) to save.  Arrays of data-type uint8 should have
        values in [0, 255], whereas floating-point arrays must be
        in [0, 1].
    format_str: str
        Format to save as, this is defaulted to PNG if using a file-like
        object; this will be derived from the extension if fname is a string
    kwargs: dict
        Keyword arguments to the Pillow save function (or tifffile save
        function, for Tiff files). These are format dependent. For example,
        Pillow's JPEG save function supports an integer ``quality`` argument
        with values in [1, 95], while TIFFFile supports a ``compress``
        integer argument with values in [0, 9].

    Notes
    -----
    Use the Python Imaging Libary.
    See PIL docs [1]_ for a list of other supported formats.
    All images besides single channel PNGs are converted using `img_as_uint8`.
    Single Channel PNGs have the following behavior:
    - Integer values in [0, 255] and Boolean types -> img_as_uint8
    - Floating point and other integers -> img_as_uint16

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    """
    # default to PNG if file-like object
    if not isinstance(fname, string_types) and format_str is None:
        format_str = "PNG"
    # Check for png in filename
    if (isinstance(fname, string_types)
            and fname.lower().endswith(".png")):
        format_str = "PNG"

    arr = np.asanyarray(arr)

    if arr.dtype.kind == 'b':
        arr = arr.astype(np.uint8)

    if arr.ndim not in (2, 3):
        raise ValueError("Invalid shape for image array: %s" % arr.shape)

    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError("Invalid number of channels in image array.")

    img = ndarray_to_pil(arr, format_str=format_str)
    img.save(fname, format=format_str, **kwargs)
