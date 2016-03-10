from ...external.tifffile import TiffFile, imsave


def imread_metadata(fname, dtype=None, **kwargs):
    """Retrieve a tiff image metadata from file, as a list
    of metadata dictionaries, one metadata dictionary for each page.

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
    with open(fname, 'rb') as f:
        tif = TiffFile(f)
        if 'key' not in kwargs:
            metadata = tif.asarray_with_metadata(**kwargs)[1]
        else:
            page = tif.pages[kwargs['key']]
            metadata = [{t: k.value for (t, k) in page.tags.items()}]
        return metadata

def imread_with_metadata(fname, dtype=None, **kwargs):
    """Load a tiff image and its metadata from file. 
    Return (image array, list of metadata dictionaries for each page).

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.
    dtype : numpy dtype object or string specifier
       Specifies data type of array elements (Not currently used).
    kwargs : keyword pairs, optional
        Additional keyword arguments to pass through (see ``tifffile``'s
        ``imread`` function).
    
    """
    if 'img_num' in kwargs:
        kwargs['key'] = kwargs.pop('img_num')
    with open(fname, 'rb') as f:
        tif = TiffFile(f)
        metadata = imread_metadata(fname, dtype, **kwargs)
        return (tif.asarray(**kwargs), metadata)

def imread(fname, dtype=None, include_metadata=False, **kwargs):
    """Load a tiff image from file. 
    Return (image array, list of metadata dictionaries for each page) 
    when return_metadata is True.
    Return image array when return_metadata is False.

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.
    dtype : numpy dtype object or string specifier
       Specifies data type of array elements (Not currently used).
    include_metadata : boolean
       Specifies whether or not to include the image metadata.
    kwargs : keyword pairs, optional
        Additional keyword arguments to pass through (see ``tifffile``'s
        ``imread`` function).
    
    """
    result = imread_with_metadata(fname, dtype, **kwargs)
    if include_metadata:
        return result
    else:
        return result[0]
