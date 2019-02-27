from skimage.data import camera
from skimage.exposure import equalize_hist
from skimage.util import img_as_float
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def quality(image: np.ndarray,
            size: int = 3) -> float:
    """
        Implementation of image quality measure based on:
        Agaian, Sos S., Karen Panetta, and Artyom M. Grigoryan.
        "A new measure of image enhancement."
        IASTED International Conference on Signal Processing
        & Communication. Citeseer, 2000.

        In its essence, it is built to quantify image quality
        based on human visual system.
    """
    image = img_as_float(image)
    ratio = np.zeros_like(image)
    if len(image.shape) > 2:
        ratio = np.divide(maximum_filter(image, (size, size, size))+1,
                          minimum_filter(image, (size, size, size))+1)
    else:
        ratio = np.divide(maximum_filter(image, (size, size))+1,
                          minimum_filter(image, (size, size))+1)
    ratio = np.mean(20*np.log(ratio))
    return ratio


def test():
    img = camera()
    print("Image quality:\n")
    print(f"\tbefore histogram equalization:{quality(img)}")
    print(f"\tafter histogram equalization:{quality(equalize_hist(img))}")


if __name__ == '__main__':
    test()
