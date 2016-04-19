from skimage.filters import gabor_kernel
from skimage import io
from matplotlib import pyplot as plt  # doctest: +SKIP

gk = gabor_kernel(frequency=0.6, aspectratio=0.5)
plt.figure()        # doctest: +SKIP
io.imshow(gk.real)  # doctest: +SKIP
io.show()           # doctest: +SKIP

