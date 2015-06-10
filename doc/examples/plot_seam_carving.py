"""
============
Seam Carving
============

This example demonstrates how images can be resized using seam carving [1]_.
Resizing often distorts contents in the image. Seam carving tries to resize
images while trying to keep important content intact. In this example we are
using the Sobel filter to signify the importance of each pixel.

.. [1] Shai Avidan and Ariel Shamir
       "Seam Carving for Content-Aware Image Resizing"
       http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf

"""
from skimage import io, data
from skimage import transform
from skimage import filters, color
from matplotlib import pyplot as plt

def nothing(img):
    return img

img = io.imread('/home/vighnesh/images/rocket.jpg')
#img = color.rgb2gray(img)
eimg = filters.sobel(color.rgb2gray(img))
#img = data.camera()
out = transform.seam_carve(img, eimg, 'vertical', 200)
#out = transform.seam_carve(out, 'horizontal', 70, energy_func=filters.sobel)
resized = transform.resize(img, out.shape)

plt.title('Original Image')
io.imshow(img, plugin='matplotlib')

#plt.figure()
#plt.title('Resized Image')
#io.imshow(resized, plugin='matplotlib')

plt.figure()
plt.title('Resized using Seam-Carving')
io.imshow(out, plugin='matplotlib')

io.show()
