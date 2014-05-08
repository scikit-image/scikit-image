__all__ = ['label']

from ..measure._ccomp import label as _label
from skimage._shared.utils import deprecated

@deprecated('skimage.measure.label')
def label(input, neighbors=8, background=None, return_num=False):
    """Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    They can be neighbors either in a 4- or 8-connected sense::

      4-connectivity      8-connectivity

           [ ]           [ ]  [ ]  [ ]
            |               \  |  /
      [ ]--[ ]--[ ]      [ ]--[ ]--[ ]
            |               /  |  \\
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int
        Whether to use 4- or 8-connectivity.
    background : int
        Consider all pixels with this value as background pixels, and label
        them as -1. (Note: background pixels will be labeled as 0 starting with
        version 0.12).
    return_num : bool
        Whether to return the number of assigned labels.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    Examples
    --------
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> print(m.label(x, neighbors=4))
    [[0 1 1]
     [2 3 1]
     [2 2 4]]

    >>> print(m.label(x, neighbors=8))
    [[0 1 1]
     [1 0 1]
     [1 1 0]]

    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])

    >>> print(m.label(x, background=0))
    [[ 0 -1 -1]
     [ 0  0  1]
     [-1 -1 -1]]

    """
    return _label(input, neighbors, background, return_num)
