import numpy as np
from numpy.testing import *
from scikits.image.transform import *

def rescale(x):
    x = x.astype(float, copy=True)
    x -= x.min()
    x /= x.max()
    return x

def test_radon_iradon():
    size = 100
    image = np.tri(size) + np.tri(size)[::-1]
    for filter_type in ["ramp", "shepp-logan", "cosine", "hamming", "hann"]:
        reconstructed = iradon(radon(image), filter=filter_type)

        image = rescale(image)
        reconstructed = rescale(reconstructed)
        delta = np.mean(np.abs(image - reconstructed))

        ## print delta
        ## import matplotlib.pyplot as plt
        ## f, (ax1, ax2) = plt.subplots(1, 2)
        ## ax1.imshow(image, cmap=plt.cm.gray)
        ## ax2.imshow(reconstructed, cmap=plt.cm.gray)
        ## plt.show()

        assert delta < 0.05

    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")
    delta = np.mean(abs(image - reconstructed))
    assert delta < 0.05
    
        
if __name__ == "__main__":
    run_module_suite()

