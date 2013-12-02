"""
========================
CenSurE feature detector
========================

The CenSurE feature detector is a scale-invariant center-surround detector
(CenSurE) that claims to outperform other detectors and is capable of real-time
implementation.

"""
from skimage import data
from skimage import transform as tf
from skimage.feature import CenSurE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img1 = rgb2gray(data.lena())
tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
                           translation=(150, -200))
img2 = tf.warp(img1, tform)

detector = CenSurE()
keypoints1, scales1 = detector.detect(img1)
keypoints2, scales2 = detector.detect(img2)

fig, ax = plt.subplots(nrows=1, ncols=2)

plt.gray()

ax[0].imshow(img1)
ax[0].axis('off')
ax[0].scatter(keypoints1[:, 1], keypoints1[:, 0], 2 ** scales1,
              facecolors='none', edgecolors='r')

ax[1].imshow(img2)
ax[1].axis('off')
ax[1].scatter(keypoints2[:, 1], keypoints2[:, 0], 2 ** scales2,
              facecolors='none', edgecolors='r')

plt.show()
