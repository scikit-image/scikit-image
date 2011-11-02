"""
=====================
GLCM Texture Features
=====================

This module provides an example of texture classification using grey
level co-occurance matrices (GLCMs). A GLCM is a histogram of 
co-occuring greyscale values at a given offset over an image. 

In this example, samples of two different textures are extracted from
an image: grassy areas and sky areas. For each patch, a GLCM with 
a horizontal offset of 5 is computed. Next, two features of the
GLCM matrices are computed: dissimilarity and correlation. These are
plotted to illustrate that the classes form clusters in feature space.

In a typical classification problem, the final step (not included in 
this example) would be to train a classifier, such as logistic 
regression, to label image patches from new images. 

"""

import os
from skimage.feature import compute_glcm, compute_glcm_prop
from skimage.io import imread
from skimage import data_dir
import matplotlib.pyplot as plt

PATCH_SIZE = 21

# open the camera image
image = imread(os.path.join(data_dir, 'camera.png'))
if False:
    plt.figure()
    plt.imshow(image)
    plt.show()
    import sys
    sys.exit()

# select some patches from grassy areas of the image
locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
grass_patches = []
for loc in locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, 
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
sky_patches = []
for loc in locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, 
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for i, patch in enumerate(grass_patches + sky_patches):
    glcm = compute_glcm(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(compute_glcm_prop(glcm, 'dissimilarity')[0, 0])
    ys.append(compute_glcm_prop(glcm, 'correlation')[0, 0])

# display the image patches
plt.figure(figsize=(8, 8))
for i, patch in enumerate(grass_patches):
    plt.subplot(3, len(grass_patches), i+1)
    plt.imshow(patch, cmap=plt.cm.gray, interpolation='nearest', 
               vmin=0, vmax=255)
    plt.xlabel('Grass %d'%(i + 1))
    
for i, patch in enumerate(sky_patches):
    plt.subplot(3, len(grass_patches), i+len(grass_patches)+1)
    plt.imshow(patch, cmap=plt.cm.gray, interpolation='nearest', 
               vmin=0, vmax=255)    
    plt.xlabel('Sky %d'%(i + 1))

# for each patch, plot (dissimilarity, correlation)
plt.subplot(3, 1, 3)
plt.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', 
         label='Grass')
plt.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo', 
         label='Sky')
plt.xlabel('GLCM Dissimilarity')
plt.ylabel('GLVM Correlation')
plt.legend()

# display the patches and plot
plt.suptitle('Grey level co-occurance matrix features', fontsize=14)
plt.show()
