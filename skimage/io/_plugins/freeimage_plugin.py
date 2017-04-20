from imageio import imread as imread_imageio, imsave as imsave_imageio
from ..._shared.utils import deprecated


@deprecated('imageio plugin')
def imread(filename):
    """
    img = imread(filename)

    Reads an image from file `filename`

    Parameters
    ----------
      filename : file name
    Returns
    -------
      img : ndarray
    """
    img = imread_imageio(filename)
    return img


@deprecated('imageio plugin')
def imsave(filename, img, **kwargs):
    '''
    imsave(filename, img)

    Save image to disk

    Image type is inferred from filename

    Parameters
    ----------
      filename : file name
      img : image to be saved as nd array

    Other parameters
    ----------------
    kwargs : keywords
        When saving as JPEG, supports the ``quality`` keyword argument which is
        an integer with values in [1, 100]
    '''
    imsave_imageio(filename, img, **kwargs)
