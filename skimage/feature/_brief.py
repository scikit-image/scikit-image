# TODO Normal sampling from image patch of size 49 x 49
# TODO Tests, example, doc

import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter

def _remove_border_keypoints(image, keypoints, dist):

	width = image.shape[0]
	height = image.shape[1]

	keypoints_list = keypoints.tolist()

	for i, j in keypoints_list:
		if i > width - dist[0] or i < dist[0] or j < dist[1] or j > height - dist[0]:
			keypoints.remove([i, j])

	keypoints = np.asarray(keypoints_list)
	return keypoints


def brief(image, keypoints, descriptor_size=256, mode='normal', patch_size=49, sample_seed=1):

	if np.squeeze(image).ndim == 3:
		image = rgb2gray(image)

	keypoints = np.round(keypoints)

	keypoints = _remove_border_keypoints(image, keypoints, (patch_size / 2, patch_size / 2))

	descriptor = np.zeros((len(keypoints), descriptor_size), dtype=int)

	# Gaussian Low pass filtering with variance 2 to alleviate noise sensitivity
	image = gaussian_filter(image, 2)

	# Sampling pairs of decision pixels in patch_size x patch_size window
	if mode == 'normal':
		np.random.seed(sample_seed)
		samples = np.round((patch_size / 5) * np.random.randn(descriptor_size * 8))
		samples = samples[samples < (patch_size / 2)]
		samples = samples[samples > - (patch_size - 1) / 2]
		first = (samples[: descriptor_size * 2]).reshape(descriptor_size, 2)
		second = (samples[descriptor_size * 2: descriptor_size * 4]).reshape(descriptor_size, 2)
	else:
		np.random.seed(sample_seed)
		samples = np.random.randint(-patch_size / 2, (patch_size / 2) + 1, (descriptor_size * 2, 2))
		first, second = np.split(samples, 2)

	for i in range(len(keypoints)):
		set_1 = first + keypoints[i]
		set_2 = second + keypoints[i]

		for j in range(descriptor_size):
			if image[set_1[j, 0]][set_1[j, 1]] < image[set_2[j, 0]][set_2[j, 0]]:
				descriptor[i][j] = 1
			else:
				descriptor[i][j] = 0

	return descriptor

def hamming_distance(descriptor_1, descriptor_2):

	distance = np.zeros((len(descriptor_1), len(descriptor_2)), dtype=int)
	for i in range(len(descriptor_1)):
		for j in range(len(descriptor_2)):
			distance[i, j] = sum(np.bitwise_xor(descriptor_1[i][:], descriptor_2[j][:]))
	return distance / descriptor_1.shape[1]
