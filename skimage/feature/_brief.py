import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import hamming


def _remove_border_keypoints(image, keypoints, dist):

	width = image.shape[0]
	height = image.shape[1]

	keypoints = keypoints[(dist < keypoints[:, 0]) & (keypoints[:, 0] < width - dist) &
				(dist < keypoints[:, 1]) & (keypoints[:, 1] < height - dist)]
	return keypoints


def brief(image, keypoints, descriptor_size=256, mode='normal', patch_size=49, sample_seed=1):
	"""Extract BRIEF Descriptor about given keypoints for a given image.

    Parameters
    ----------
    image : ndarray
        Input image.
    keypoints : (P, 2) ndarray
        Array of keypoint locations.
    descriptor_size : int
        Size of BRIEF descriptor about each keypoint. Sizes 128, 256 and 512
        preferred by the authors. Default is 256.
    mode : string
        Probability distribution for sampling location of decision pixel-pairs
        around keypoints. Default is 'normal' otherwise uniform.
    patch_size : int
        Length of the two dimensional square patch sampling region around
        the keypoints. Default is 49.
    sample_seed : int
        Seed for sampling the decision pixel-pairs. Default is 1.

    Returns
    -------
    descriptor : ndarray with dtype bool
        2D ndarray of dimensions (no_of_keypoints, descriptor_size) with value
        at an index (i, j) either being True or False representing the outcome
        of Intensity comparison about ith keypoint on jth decision pixel-pair.

    References
    ----------
    .. [1] Michael Calonder, Vincent Lepetit, Christoph Strecha and Pascal Fua
    "BRIEF : Binary robust independent elementary features",
    http://cvlabwww.epfl.ch/~lepetit/papers/calonder_eccv10.pdf

    """
	if np.squeeze(image).ndim == 3:
		image = rgb2gray(image)

	keypoints = np.array(keypoints + 0.5, dtype=np.intp)

	# Removing keypoints that are (patch_size / 2) distance from the image border
	keypoints = _remove_border_keypoints(image, keypoints, patch_size / 2)

	descriptor = np.zeros((len(keypoints), descriptor_size), dtype=bool)

	# Gaussian Low pass filtering with variance 2 to alleviate noise sensitivity
	image = gaussian_filter(image, 2)

	# Sampling pairs of decision pixels in patch_size x patch_size window
	if mode == 'normal':
		np.random.seed(sample_seed)
		samples = np.round((patch_size / 5) * np.random.randn(descriptor_size * 8))
		samples = samples[(samples < (patch_size / 2)) & (samples > - (patch_size - 1) / 2)]
		first = (samples[: descriptor_size * 2]).reshape(descriptor_size, 2)
		second = (samples[descriptor_size * 2: descriptor_size * 4]).reshape(descriptor_size, 2)
	else:
		np.random.seed(sample_seed)
		samples = np.random.randint(-patch_size / 2, (patch_size / 2) + 1, (descriptor_size * 2, 2))
		first, second = np.split(samples, 2)

	# Intensity comparison tests for building the descriptor
	for i in range(len(keypoints)):
		set_1 = first + keypoints[i]
		set_2 = second + keypoints[i]

		for j in range(descriptor_size):
			if image[set_1[j, 0]][set_1[j, 1]] < image[set_2[j, 0]][set_2[j, 0]]:
				descriptor[i][j] = True

	return descriptor


def hamming_distance(descriptor_1, descriptor_2):
	"""A dissimilarity measure used for matching keypoints in different images
	using binary feature descriptors like BRIEF etc.

    Parameters
    ----------
    descriptor_1 : ndarray with dtype bool
    	Binary feature descriptor for keypoints in the first image.
    	2D ndarray of dimensions (no_of_keypoints_in_image_1, descriptor_size)
    	with value at an index (i, j) either being True or False representing
    	the outcome of Intensity comparison about ith keypoint on jth decision
    	pixel-pair.
    descriptor_2 : ndarray with dtype bool
    	Binary feature descriptor for keypoints in the second image.
    	2D ndarray of dimensions (no_of_keypoints_in_image_2, descriptor_size)
    	with value at an index (i, j) either being True or False representing
    	the outcome of Intensity comparison about ith keypoint on jth decision
    	pixel-pair.

    Returns
    -------
    distance : ndarray
    	2D ndarray of dimensions (no_of_rows_in_descripto_1, no_of_rows_in_descripto_2)
    	with value at an index (i, j) between the range [0, 1] representing the
    	extent of dissimilarity between ith keypoint of in first image and jth
    	keypoint in second image.

    """
	distance = np.zeros((len(descriptor_1), len(descriptor_2)), dtype=float)
	for i in range(len(descriptor_1)):
		for j in range(len(descriptor_2)):
			distance[i, j] = hamming(descriptor_1[i][:], descriptor_2[j][:])
	return distance
