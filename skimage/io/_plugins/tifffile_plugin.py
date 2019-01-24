from ...external.tifffile import TiffFile, imsave, parse_kwargs


def imread(fname, **kwargs):
    """Load a tiff image from file.

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.
    dtype : numpy dtype object or string specifier
       Specifies data type of array elements (Not currently used).
    kwargs : keyword pairs, optional
        Additional keyword arguments to pass through (see ``tifffile``'s
        ``imread`` function).

    Notes
    -----
    Provided by Christophe Golhke's tifffile.py [1]_, and supports many
    advanced image types including multi-page and floating point.

    References
    ----------
    .. [1] http://www.lfd.uci.edu/~gohlke/code/tifffile.py

    """

    if 'img_num' in kwargs:
        kwargs['key'] = kwargs.pop('img_num')

    # separate TiffFile kwargs from other kwargs
    tiff_keys = ['multifile', 'multifile_close', 'pages', 'fastij', 'is_ome',
                 'pattern']
    kwargs_tiff = parse_kwargs(kwargs, *tiff_keys)
    for key in tiff_keys:
        kwargs.pop(key, None)

    # read and return tiff as numpy array
    with TiffFile(fname, **kwargs_tiff) as tif:
        return tif.asarray(**kwargs)
