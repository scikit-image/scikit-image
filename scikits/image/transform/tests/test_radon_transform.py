import numpy as np
from numpy.testing import *
from scikits.image.transform import *


def test_radon_iradon():
    size = 100
    image = np.tri(size) + np.tri(size)[::-1]
    for filter_type in ["ramp", "shepp-logan", "cosine", "hamming", "hann"]:
        reconstructed = iradon(radon(image), filter=filter_type)
        delta = np.sum(abs(image/np.max(image) - reconstructed/np.max(reconstructed)))/(size*size)
        assert delta < 0.1
    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")
    delta = np.sum(abs(image/np.max(image) - reconstructed/np.max(reconstructed)))/(size*size)
    assert delta < 0.1
    
        
if __name__ == "__main__":
    run_module_suite()

