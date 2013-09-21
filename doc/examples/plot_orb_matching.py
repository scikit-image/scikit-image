import numpy as np
from skimage import data
from skimage import transform as tf
from skimage.feature import pairwise_hamming_distance, brief, match_binary_descriptors, corner_harris, corner_peaks, keypoints_orb, descriptor_orb
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt

rotate = 0.5
translate = (-100, -200)
scaling = (1.5, 1.5)
match_threshold = 0.40
match_cross_check = True

img_color = data.lena()
tform = tf.AffineTransform(scale = scaling, rotation=rotate, translation=translate)
transformed_img_color = tf.warp(img_color, tform)
img = rgb2gray(img_color)
transformed_img = rgb2gray(transformed_img_color)

keypoints1, orientations1, scales1 = keypoints_orb(img, n_keypoints=250)
keypoints1.shape
descriptors1, keypoints1 = descriptor_orb(img, keypoints1, orientations1, scales1)
keypoints1.shape
descriptors1.shape

keypoints2, orientations2, scales2 = keypoints_orb(transformed_img, n_keypoints=250)
keypoints2.shape
descriptors2, keypoints2 = descriptor_orb(transformed_img, keypoints2, orientations2, scales2)
keypoints2.shape
descriptors2.shape

pairwise_hamming_distance(descriptors1, descriptors2)
matched_keypoints, mask1, mask2 = match_binary_descriptors(keypoints1, descriptors1, keypoints2, descriptors2, cross_check=match_cross_check, threshold=match_threshold)

matched_keypoints.shape

# Plotting the matched correspondences in both the images using matplotlib
src = matched_keypoints[:, 0, :]
dst = matched_keypoints[:, 1, :]
src_scale = 10 * (scales1[mask1] + 1) ** 2
dst_scale = 10 * (scales2[mask2] + 1) ** 2

img_combined = np.concatenate((img_as_float(img_color), img_as_float(transformed_img_color)), axis=1)
offset = img.shape

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.gray()

ax.imshow(img_combined, interpolation='nearest')
ax.axis('off')
ax.axis((0, 2 * offset[1], offset[0], 0))
ax.set_title('Matched correspondences : Rotation = %f; Scale = %s; Translation = %s; threshold = %f; cross_check = %r' % (rotate, scaling, translate, match_threshold, match_cross_check))

for m in range(len(src)):
    ax.plot((src[m, 1], dst[m, 1] + offset[1]), (src[m, 0], dst[m, 0]), '-', color='g')
    ax.scatter(src[m, 1], src[m, 0], src_scale[m], facecolors='none', edgecolors='b')
    ax.scatter(dst[m, 1] + offset[1], dst[m, 0], dst_scale[m], facecolors='none', edgecolors='b')

plt.show()
