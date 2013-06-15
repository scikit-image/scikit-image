# TODO Normal sampling from image patch of size 49 x 49
# TODO Tests, example, doc

import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter

KERNEL_SIZE = (9, 9)
PATCH_SIZE = (49, 49)


def _remove_border_keypoints(image, keypoints, dist):

	width = image.shape[0]
	height = image.shape[1]
	for i, j in keypoints:
		if i > width - dist[0] or i < dist[0] or j < dist[1] or j > height - dist[0]:
			keypoints.remove((i, j))
	return keypoints


def brief(image, keypoints, descriptor_size=32, mode='uniform'):

	if descriptor_size not in (16, 32, 64):
		raise ValueError('Descriptor size should be either 16, 32 or 64 bytes')

	if np.squeeze(image).ndim == 3:
		image = rgb2gray(image)

	keypoints = _remove_border_keypoints(image, keypoints, (PATCH_SIZE[0] / 2, PATCH_SIZE[1] / 2))

	descriptor = np.zeros((len(keypoints), descriptor_size * 8), dtype=int)

	image = gaussian_filter(image, 2)

	if mode == 'uniform':
		np.random.seed(1)
		first = np.random.randint(-24, 25, (descriptor_size * 8, 2))
		np.random.seed(2)
		second = np.random.randint(-24, 25, (descriptor_size * 8, 2))
	else:
		#TODO mode='normal'
		pass

	for i in range(len(keypoints)):
		set_1 = first + keypoints[i]
		set_2 = second + keypoints[i]

		for j in range(descriptor_size * 8):
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
