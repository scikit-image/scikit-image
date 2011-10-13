from scikits.image.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((400, 400))
image[10:-10, 10:100] = 1
image[-100:-10, 10:-10] = 1
image[10:-10, -100:-10] = 1

skeleton = skeletonize.skeletonize(image)

plt.figure(figsize=(8,5))

plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original', fontsize=20)

plt.subplot(122)
plt.imshow(skeleton, cmap=plt.cm.gray)
plt.axis('off')
plt.title('skeleton', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98, 
                    bottom=0.02, left=0.02, right=0.98)

plt.show()