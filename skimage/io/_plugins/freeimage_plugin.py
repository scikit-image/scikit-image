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
def imsave(filename, img):
    '''
    imsave(filename, img)

    Save image to disk

    Image type is inferred from filename

    Parameters
    ----------
      filename : file name
      img : image to be saved as nd array
    '''
    imsave_imageio(filename, img)
