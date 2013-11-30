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


def plot_matches(ax, image1, image2, keypoints1, keypoints2,
                 indices1, indices2, keypoints_color='k', matches_color=None,
                 only_matches=False):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    keypoints : (K1, 2) array
        Keypoint coordinates as ``(row, col)``.
    indices1 : (Q, ) array
        Indices of corresponding matches for first set of keypoints.
    indices2 : (Q, ) array
        Indices of corresponding matches for second set of keypoints.
    keypoints_color : matplotlib color
        Color for keypoint locations.
    matches_color : matplotlib color
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool
        Whether to only plot matches and not plot the keypoint locations.

    """

    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    indices1 = np.squeeze(indices1)
    indices2 = np.squeeze(indices2)

    new_shape1 = image1.shape
    new_shape2 = image2.shape

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image)
    ax.axis((0, 2 * offset[1], offset[0], 0))

    for i in range(len(indices1)):
        idx1 = indices1[i]
        idx2 = indices2[i]

        if matches_color is None:
            color = np.random.rand(3, 1)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),
                '-', color=color)


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
    keypoints : (N, 2) array
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
