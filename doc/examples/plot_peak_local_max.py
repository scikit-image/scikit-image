"""
===============================================================================
Peak local maximum
===============================================================================

The peak local maximum return coordinates of peaks in a image. The maximum
filter is used for finding the maximum peaks in the image. It dilates the 
original image and is used within peak local max function to find the 
coordinates of maximum peaks, comparing the dilated image with the original. 
Then, the peak local max function returns the coordinates of points where 
image = dilated image. 

"""
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from skimage import data, img_as_float

im = img_as_float(data.coins())
image = im.copy()

# The image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function 
image_max = ndimage.maximum_filter(image, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of maximum peaks
coordinates = peak_local_max(im, min_distance = 20)

# display results
plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Original')

plt.subplot(132)
plt.imshow(image_max, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Maximum filter')

plt.subplot(133)
plt.imshow(im, cmap=plt.cm.gray)
a, b = im.shape
plt.plot([p[1] for p in coordinates],[p[0] for p in coordinates],'r.')
plt.xlim(0,b)
plt.ylim(a,0)
plt.axis('off')
plt.title('Peak local max')

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

