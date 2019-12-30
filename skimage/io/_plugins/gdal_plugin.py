__all__ = ['imread']

from warnings import warn

try:
    import osgeo.gdal as gdal
except ImportError:
    raise ImportError("The GDAL Library could not be found. "
                      "Please refer to http://www.gdal.org/ "
                      "for further instructions.")


def imread(fname, dtype=None):
    """Load an image from file.

    """
    if dtype is not None:
        warn('The dtype argument was always silently ignored. It will be '
             'removed from scikit-image version 0.17. To avoid this '
             'warning, do not specify it in your function call.',
             UserWarning, stacklevel=2)

    ds = gdal.Open(fname)

    return ds.ReadAsArray().astype(dtype)
