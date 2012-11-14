"""
===================
Entropy
===================


"""
from skimage import data
from skimage.filter.rank import entropy
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt

# defining a 8- and a 16-bit test images
a8 = data.camera()
a16 = data.camera().astype(np.uint16)*4

ent8 = entropy(a8,disk(5)) # pixel value contain 10x the local entropy
ent16 = entropy(a16,disk(5)) # pixel value contain 1000x the local entropy

# display results
plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.imshow(a8, cmap=plt.cm.gray)
plt.xlabel('8-bit image')
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(ent8, cmap=plt.cm.jet)
plt.xlabel('entropy*10')
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(a16, cmap=plt.cm.gray)
plt.xlabel('16-bit image')
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(ent16, cmap=plt.cm.jet)
plt.xlabel('entropy*1000')
plt.colorbar()
plt.show()

