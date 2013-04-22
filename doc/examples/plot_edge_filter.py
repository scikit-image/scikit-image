import matplotlib
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filter import roberts,sobel

image = camera()
edge_roberts = roberts(image)
edge_sobel = sobel(image)

plt.figure(figsize=(8, 2.5))
plt.subplot(1, 2, 1)
plt.imshow(edge_roberts, cmap=plt.cm.gray)
plt.title('Roberts Edge Detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_sobel, cmap=plt.cm.gray)
plt.title('Sobel Edge Detection')
plt.axis('off')


plt.show()
