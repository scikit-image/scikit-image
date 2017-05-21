"""
============
Steerable Pyramid
============

Steerable pyramid decomposition 
.. [1] http://http://www.cns.nyu.edu/~eero/steerpyr/

"""
import matplotlib
import matplotlib.pyplot as plt

from skimage.data import coffee
from skimage.color import rgb2gray
from skimage.transform import steerable


matplotlib.rcParams['font.size'] = 9


image = coffee()
image = rgb2gray(image)
coeff = steerable.build_steerable(image)

height = len(coeff)
fig, axes = plt.subplots(height, 4, figsize=(8, 7))
axes[0, 0].imshow(coeff[0])
axes[height - 1, 0].imshow(coeff[-1])

for i in range(1, height - 1):
    for j in range(4):
        axes[i][j].imshow(coeff[i][j])

plt.show()
