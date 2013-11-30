import numpy as np

from skimage.util import img_as_float


class FeatureDetector(object):

    def detect(self, image):
        """Detect keypoints in image.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        raise NotImplementedError()


class DescriptorExtractor(object):

    def extract(self, image, keypoints):
        """Extract feature descriptors in image for given keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint locations as ``(row, col)``.

        """
        raise NotImplementedError()


def _prepare_grayscale_input_2D(image):
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    return img_as_float(image)


def _mask_border_keypoints(image_shape, keypoints, distance):
    """Mask coordinates that are within certain distance from the image border.

    Parameters
    ----------
    image_shape : (2, ) array_like
        Shape of the image as ``(rows, cols)``.
    coords : (N, 2) array
        Keypoint coordinates as ``(rows, cols)``.
    distance : int
        Image border distance.

    Returns
    -------
    mask : (N, ) bool array
        Mask indicating if pixels are within the image (``True``) or in the
        border region of the image (``False``).

    """

    rows = image_shape[0]
    cols = image_shape[1]

    mask = (((distance - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (rows - distance + 1))
            & ((distance - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (cols - distance + 1)))

    return mask
