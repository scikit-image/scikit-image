"""
==============
Shape Contexts
==============

The Shape Context descriptor was introduced by Serge Belongie and Jitendra
Malik.

The Shape Context descriptor captures the coarse distribution of the rest
of the shape with respect to a given point on the shape.
It is the log-polar histogram of points on the shape relative
to the given point. It uses bins that are uniform in log-polar space
to make the descriptor more sensitive to positions of nearby sample points than
to those of points farther away.
Corresponding points on two similar shapes will have similar shape contexts.
Finding correspondences between two shapes is equivalent to finding for each
point on one shape the sample point on the other shape that has the most
similar shape context, which can be solved as an optimal assignment problem.

This examples is an illustration to show that corresponding points on two
similar shapes have similar shape contexts Point 1 and Point 2 annotated on
shapes are corresponding points. Their shape contexts will be similar, whereas
the shape context of Point 3 will be quite different from those of Point 1 and Point 2.

See their paper_ for further details.

.. _paper: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf

    References
    ----------
    .. [1]  Serge Belongie, Jitendra Malik and Jan Puzicha.
            "Shape matching and object recognition using shape contexts."
            IEEE PAMI 2002.
"""
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.feature import shape_context
from skimage.filter import canny
from skimage.transform import resize
from skimage.io import imread
from skimage.data import data_dir


fig = plt.figure(figsize=(10, 10))

img1 = imread(data_dir + '/first_9.png')
img1 = resize(img1, (70, 70), order=1)
ax = fig.add_subplot(3, 2, 1)
ax.imshow(img1, cmap=plt.cm.gray_r)
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('First 9')

img1_edges = canny(img1)
ax = fig.add_subplot(3, 2, 2)
ax.imshow(img1_edges, cmap=plt.cm.gray_r)
pt1 = (47, 45)
pt3 = (25, 44)
ax.plot(pt1[0], pt1[1], 'gs-')
ax.plot(pt3[0], pt3[1], 'ro-')
ax.annotate('Point 1', xy=pt1, xytext=(48, 65),
            arrowprops=dict(arrowstyle='->'))
ax.annotate('Point 3', xy=pt3, xytext=(10, 60),
            arrowprops=dict(arrowstyle='->'))
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Shape First 9')

img2 = imread(data_dir + '/second_9.png')
img2 = resize(img2, (70, 70), order=1)
ax = fig.add_subplot(3, 2, 3)
ax.imshow(img2, cmap=plt.cm.gray_r)
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Second 9')

img2_edges = canny(img2)
pt2 = (45, 44)
ax = fig.add_subplot(3, 2, 4)
ax.imshow(img2_edges, cmap=plt.cm.gray_r)
ax.plot(pt2[0], pt2[1], 'gd-')
ax.annotate('Point 2', xy=pt2, xytext=(48, 60),
            arrowprops=dict(arrowstyle='->'))
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Shape Second 9')


pt1_histogram = shape_context(img_as_float(img1_edges), pt1, 0, 25,
                              radial_bins=5, polar_bins=12)
ax = fig.add_subplot(3, 3, 7)
ax.imshow(pt1_histogram, cmap=plt.cm.gray_r, interpolation='nearest')
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_ylabel('log r')
ax.set_xlabel(r'$\Theta$')
ax.set_title('Shape context of Point 1')

pt2_histogram = shape_context(img_as_float(img2_edges), pt2, 0, 25,
                              radial_bins=5, polar_bins=12)
ax = fig.add_subplot(3, 3, 8)
ax.imshow(pt2_histogram, cmap=plt.cm.gray_r, interpolation='nearest')
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_ylabel('log r')
ax.set_xlabel(r'$\Theta$')
ax.set_title('Shape context of Point 2')


pt3_histogram = shape_context(img_as_float(img1_edges), pt3, 0, 25,
                              radial_bins=5, polar_bins=12)
ax = fig.add_subplot(3, 3, 9)
ax.imshow(pt3_histogram, cmap=plt.cm.gray_r, interpolation='nearest')
ax.axis('image')
ax.set_xticks(())
ax.set_yticks(())
ax.set_ylabel('log r')
ax.set_xlabel(r'$\Theta$')
ax.set_title('Shape context of Point 3')

fig.suptitle('Shape Contexts', fontsize=20)
plt.show()
