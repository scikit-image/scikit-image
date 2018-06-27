try:
    import osgeo.gdal as gdal
except ImportError:
    raise ImportError("The GDAL Library could not be found. "
                      "Please refer to http://www.gdal.org/ "
                      "for further instructions.")

__all__ = ['imread']


def imread(fname, dtype=None):
    """Load an image from file.

    """
    ds = gdal.Open(fname)

    return ds.ReadAsArray().astype(dtype)
