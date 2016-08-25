"""
=====================
FastDRaW segmentation
=====================

The FastDRaW algorithm [1]_ is based on the Random Walker (RW) algorithm [2]_
to perform fast image segmentation using a set of pixels labeled as belonging 
to different objects (e.g., foreground and background labels). The 
computational bottleneck for RW is solving a linear system whose size increases
with the image size. To reduce the computation time, the FastDRaW algorithm
relies on a coarse-to-fine segmentation strategy, performed in two steps:
First, on a low-resolution version of the image, a coarse RW segmentation is
computed over a restricted region of interest (ROI). Second, the result is 
refined by applying the RW algorithm at full resolution over a narrow strip
around the coarse contour. 

Note: Where RW computes the probabilities of pixels belonging to each label
category, FastDRaW computes the probabilities of pixels belonging to a target
label category. For example, if the image contains labels [1, 2, 3] and the
target label is 1, then, both labels 2 and 3 are considered outside the object,
when label 1 is considered as belonging to the object.

Pros:
- FastDRaW allows a major gain in computation time
- Implicit labeling of pixels lying outside the ROI as not belonging to the 
  target label category, which reduces the required amount of labels to 
  acheive the segmentation
Cons:
- The sum of probabilities of each label category is no longer ensured to be
  equal to 1. This is due to the target label strategy which forces the pixels
  outside the ROI to have zero probability belonging to the target label.
  
In this example, we create an interactive window to compare between RW and 
FastDRaW segmetnations in the case of two label categories (foreground and
background). The user labels the original image (at left) using the mouse:
left click for foreground and right click for background. Then, using "space
bar" to run both segmentation. The computation time is displayed.

.. [1] H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte,
    *FastDRaW - Fast Delineation by Random Walker: application to large
    images*, MICCAI Workshop on Interactive Medical Image Computing (IMIC),
    Athens, Greece, (2016).
.. [2] Leo Grady, *Random walks for image segmentation*, IEEE Trans. Pattern
       Anal. Mach. Intell. 2006 Nov; 28(11):1768-83

"""

import numpy as np
import matplotlib.pyplot as plt
import time

from skimage import data
from skimage.morphology import dilation, disk
from skimage.segmentation import random_walker
from skimage.segmentation import FastDRaW

"""
Callback functions
------------------
"""
# drawing callback function
def onDraw(event):
    """This function is called when the mouse moves over the image
    """
    if event.button == 1:
        # if left click is detected during the mouse mouvement
        # get x and y coordinates
        yy = int(event.ydata)
        xx = int(event.xdata)
        # draw red pixels on the rgb image
        rgb_image[yy-brush : yy+brush+1, xx-brush : xx+brush+1] = (255,0,0)
        # generate labels as foreground, 1
        labels[yy-brush : yy+brush+1, xx-brush : xx+brush+1] = 1
        # refresh plot
        pltimg.set_data(rgb_image)
        plt.draw()
    elif event.button == 3:
        # if right click is detected during the mouse mouvement
        # get x and y coordinates
        yy = int(event.ydata)
        xx = int(event.xdata)
        # draw green pixels on the rgb image
        rgb_image[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = (0,255,0)
        # generate labels as background, 2
        labels[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = 2
        # refresh plot
        pltimg.set_data(rgb_image)
        plt.draw()

# one click callback function
def onClick(event):
    """This function is called when mouse buttons are clicked
    """
    # get x and y coordinates
    yy = int(event.ydata)
    xx = int(event.xdata)
    if event.button == 1:
        # draw red pixels on the rgb image
        rgb_image[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = (255,0,0)
        # generate labels as foreground, 1
        labels[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = 1
    elif event.button == 3:
        # draw green pixels on the rgb image
        rgb_image[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = (0,255,0)
        # generate labels as background, 2
        labels[yy-brush:yy+brush+1,xx-brush:xx+brush+1] = 2
    # refresh plot
    pltimg.set_data(rgb_image)
    plt.draw()
    
# keyboard callback events
def onKeypress(event):
    """This function is called when keyboard buttons are pressed
    """
    global target_label
    if event.key == ' ':
        # if space bar is pressed, FastDRaW  and RW segmentations are performed
        
        # FastDRaW
        t = time.clock()
        # --- perform the FastDRaW segmentation
        segm = fastdraw.update(labels, target_label=target_label, k=0.5)
        # --- display the contour in red over the original image
        contour = (dilation(segm, disk(1)) - segm).astype(np.bool)
        result = original_image.copy()
        result[contour,:] = (255,0,0)
        # --- show the image
        ax2.imshow(result)
        ax2.set_title("FastDRaW time "+str(time.clock()-t)+" s")
        plt.draw()
        
        # RW
        t = time.clock()
        # perform the RW segmentation
        segm = random_walker(image,labels,beta=beta)
        # select which label to display
        # since this is a binary case segmentation (foreground/background),
        # segm == 1 is similar to segm == 2
        segm = segm == 1 
        # --- display the contour in red over the original image
        contour = (dilation(segm, disk(1)) - segm).astype(np.bool)
        result = original_image.copy()
        result[contour,:] = (255,0,0)
        # --- show the image
        ax3.imshow(result)
        ax3.set_title("RW time "+str(time.clock()-t)+" s")
        plt.draw()
    

"""
Main
----
"""
# load array like image
image = data.coins()
# convert image to rgb for drawing purposes
if image.ndim == 2:
    rgb_image = np.dstack((image,image,image))
# create a copy of the rgb image to display the results
original_image = rgb_image.copy()
# initialize labels to zeros
labels = np.zeros_like(image)

# display the original image, on which the user can draw
fig = plt.figure()
ax1 = fig.add_subplot(131)
pltimg = plt.imshow(rgb_image)
ax1.set_title('Original image')
plt.axis('off')

# set callback functions
did = fig.canvas.mpl_connect('motion_notify_event', onDraw)
cid = fig.canvas.mpl_connect('button_press_event', onClick)
kid = fig.canvas.mpl_connect('key_press_event', onKeypress)

# display FastDRaW result image
ax2 = fig.add_subplot(132)
ax2.imshow(original_image)
ax2.set_title('FastDRaW')
plt.axis('off')

# display Random Walker result image
ax3 = fig.add_subplot(133)
ax3.imshow(original_image)
ax3.set_title('RW')
plt.axis('off')

# size (in pixels) of the drawing brush
brush = 3
# set beta to 100 (default 300 for FastDRaW and 130 for RW)
# for comparison purposes, beta is set to the same value
beta = 100
# set target label category to 1
target_label = 1
# create an instance of `FastDRaW` (optional: beta and down-sampled image size)
fastdraw = FastDRaW(image, beta=beta, downsampled_size=100)

plt.show()
