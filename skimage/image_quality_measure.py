from skimage.data import camera
from skimage.exposure import equalize_hist
from skimage.util import img_as_float
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def quality(image: np.ndarray,
            size: int = 3) -> float:
    """
        The image quality measure called EME based on [1]_.

        Parameters
        ----------
        image : array
            Input image of which the quality should be assessed.
            Can be either 3-channel RGB or 1-channel grayscale.
            The function converts pixel intensities into floats by default.
        size : int
            Size of the window. Default value is 3.

        Returns
        -------
        eme : float
            The number describing image quality.

        References
        ----------
        .. [1] Agaian, Sos S., Karen Panetta, and Artyom M. Grigoryan.
               "A new measure of image enhancement."
               IASTED International Conference on Signal Processing
               & Communication. Citeseer, 2000.
        Examples
        --------
        >>> from skimage.data import camera
        >>> from skimage.exposure import equalize_hist
        >>> img = camera()
        >>> print("Image quality:\n")
            Image quality:
        >>> print(f"\tbefore histogram equalization: {quality(img)}")
                before histogram equalization: 0.9096745071475523
        >>> print(f"\tafter histogram equalization:{quality(equalize_hist(img))}")
                after histogram equalization: 1.299327371881219

    """
    image = img_as_float(image)
    eme = np.zeros_like(image)
    if len(image.shape) > 2:
        eme = np.divide(maximum_filter(image, (size, size, size))+1,
                        minimum_filter(image, (size, size, size))+1)
    else:
        eme = np.divide(maximum_filter(image, (size, size))+1,
                        minimum_filter(image, (size, size))+1)
    eme = np.mean(20*np.log(eme))
    return eme
