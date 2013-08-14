"""
===========================
Censure Keypoints Detection
===========================

In this example, we detect and plot the Censure Keypoints at various scales
using Difference of Boxes, Octagon and Star shaped bi-level filters.

"""
from skimage.feature import keypoints_censure
from skimage.data import lena
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Initializing the parameters for Censure keypoints
img = lena()
gray_img = rgb2gray(img)
min_scale = 1
max_scale = 7
nms_threshold = 0.15
rpc_threshold = 10

# Detecting Censure keypoints for the following filters
for mode in ['dob', 'octagon', 'star']:

    kp_censure, scale = keypoints_censure(gray_img, min_scale, max_scale,
                                          mode, nms_threshold, rpc_threshold)
    f, axarr = plt.subplots((max_scale - min_scale + 1) // 3, 3)

    # Plotting Censure features at all the scales
    for i in range(max_scale - min_scale - 1):
        keypoints = kp_censure[scale == i + min_scale + 1]
        num = len(keypoints)
        x = keypoints[:, 1]
        y = keypoints[:, 0]
        s = 5 * 2**(i + min_scale + 1)
        axarr[i // 3, i - (i // 3) * 3].imshow(img)
        axarr[i // 3, i - (i // 3) * 3].scatter(x, y, s, facecolors='none',
                                                edgecolors='g')
        axarr[i // 3, i - (i // 3) * 3].set_title(' %s %s-Censure features at '
                                                  'scale %d' % (num, mode, i +
                                                                min_scale + 1))

    plt.suptitle('NMS threshold = %f, RPC threshold = %d'
                 % (nms_threshold, rpc_threshold))
plt.show()
