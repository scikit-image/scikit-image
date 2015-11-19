"""
============================
Robust matching using RANSAC
============================

In this simplified example we first generate two synthetic images as if they
were taken from different view points.

In the next step we find interest points in both images and find
correspondences based on a weighted sum of squared differences of a small
neighborhood around them. Note, that this measure is only robust towards
linear radiometric and not geometric distortions and is thus only usable with
slight view point changes.

After finding the correspondences we end up having a set of source and
destination coordinates which can be used to estimate the geometric
transformation between both images. However, many of the correspondences are
faulty and simply estimating the parameter set with all coordinates is not
sufficient. Therefore, the RANSAC algorithm is used on top of the normal model
to robustly estimate the parameter set by detecting outliers.

"""
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage  import io

original=True

# warp synthetic image
tform = AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))
    
if original:
    # generate synthetic checkerboard image and add gradient for the later matching
    checkerboard = img_as_float(data.checkerboard())
    img_orig = np.zeros(list(checkerboard.shape) + [3])
    img_orig[..., 0] = checkerboard
    gradient_r, gradient_c = (np.mgrid[0:img_orig.shape[0],
                                       0:img_orig.shape[1]]
                              / float(img_orig.shape[0]))
    img_orig[..., 1] = gradient_r
    img_orig[..., 2] = gradient_c
    img_orig = rescale_intensity(img_orig)
    img_orig_gray = rgb2gray(img_orig)
    
    img_warped = warp(img_orig, tform.inverse, output_shape=(200, 200))
    img_warped_gray = rgb2gray(img_warped)
else:
    imageName='test1.png'
    imageName2='test2.png'    
    
    img1  = io.imread(imageName)
    img2  = io.imread(imageName2)
    
    img_orig        = img1
    img_warped      = img2
    img_orig_gray   = rgb2gray(img_orig)
    img_warped_gray = rgb2gray(img_warped)
    

# extract corners using Harris' corner measure
coords_orig = corner_peaks(corner_harris(img_orig_gray), threshold_rel=0.001,
                           min_distance=5)
coords_warped = corner_peaks(corner_harris(img_warped_gray),
                             threshold_rel=0.001, min_distance=5)

# determine sub-pixel corner position
coords_orig_subpix = corner_subpix(img_orig_gray, coords_orig, window_size=9)
coords_warped_subpix = corner_subpix(img_warped_gray, coords_warped,
                                     window_size=9)


def gaussian_weights(patch_radius, sigma=1):
    y, x = np.mgrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


# find correspondences using simple weighted sum of squared differences
def match_corner(coord_list1,coord_list2,img1,img2, patch_radius=5):
    c1len = len(coord_list1)
    c2len = len(coord_list2)
    cl1index = np.arange(c1len)
    cl2index = np.arange(c2len)
    src = []
    dst = []
    for [[r,c],idx] in zip(coord_list1,cl1index):
        src.append(idx)
        window_orig = img1[r-patch_radius:r+patch_radius+1,
                          c-patch_radius:c+patch_radius+1]
    
        # weight pixels depending on distance to center pixel
        weights = gaussian_weights(patch_radius, 3)
    
        # compute sum of squared differences to all corners in warped image
        SSDs = []
        for cr, cc in coord_list2:
            window_warped = img2[cr-patch_radius:cr+patch_radius+1,
                                 cc-patch_radius:cc+patch_radius+1]
            SSD = np.sum(weights * (window_orig - window_warped)**2)
            SSDs.append(SSD)
    
        # use corner with minimum SSD as correspondence
        min_idx = np.argmin(SSDs)
        dst.append(min_idx)
    
   
    return [np.array(coord_list1[src]),np.array(coord_list2[dst])]


# find correspondences using simple weighted sum of squared differences
[src,dst] = match_corner(coords_orig,coords_warped,img_orig_gray,img_warped_gray, patch_radius=5)



# estimate affine transform model using all coordinates
model = AffineTransform()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=8,
                               residual_threshold=2, max_trials=10000)
outliers = inliers == False


# compare "true" and estimated transform parameters
print(tform.scale, tform.translation, tform.rotation)
print(model.scale, model.translation, model.rotation)
print(model_robust.scale, model_robust.translation, model_robust.rotation)

# bug workaround swapping X,Y
# issue: https://github.com/scikit-image/scikit-image/issues/1789
model_robust.params[[0, 1], :] = model_robust.params[[1, 0], :]

# visualize correspondence
fig, ax = plt.subplots(nrows=2, ncols=2)

def warp_and_combine(img1,img2,xform):
    img1_warped = warp(img1,xform)
    img1f = img1_warped.astype(float)*0.5
    img2f = img_warped_gray.astype(float)*0.5
    if img1f.shape == img2f.shape:
        combined = img1f+img2f
    else:
        combined = img1f
    
    return combined
        
plt.gray()

inlier_idxs = np.nonzero(inliers)[0]
plot_matches(ax[0,0], img_orig_gray, img_warped_gray, src, dst,
             np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
ax[0,0].axis('off')
ax[0,0].set_title('Correct Correspondences')

outlier_idxs = np.nonzero(outliers)[0]
plot_matches(ax[0,1], img_orig_gray, img_warped_gray, src, dst,
             np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
ax[0,1].axis('off')
ax[0,1].set_title('Faulty Correspondences')

ax[1,0].axis('off')
ax[1,0].set_title('Combine Originals')
ax[1,0].imshow(warp_and_combine(img_orig_gray,img_warped_gray,AffineTransform()))


ax[1,1].axis('off')
ax[1,1].set_title('Warp then Combine')
ax[1,1].imshow(warp_and_combine(img_orig_gray,img_warped_gray,model_robust.inverse))

plt.show()
