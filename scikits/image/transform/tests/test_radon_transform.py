import numpy as np
from numpy.testing import *
from scikits.image.transform import *

def rescale(x):
    x = x.astype(float)
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
   
def test_iradon_angles():
    """
    Test with different number of projections
    """
    size = 100
    # Synthetic data
    image = np.tri(size) + np.tri(size)[::-1]
    # Large number of projections: a good quality is expected
    nb_angles = 200
    radon_image_200 = radon(image, theta=np.linspace(0, 180, nb_angles, 
                    endpoint=False))
    reconstructed = iradon(radon_image_200)
    delta_200 = np.mean(abs(rescale(image) - rescale(reconstructed)))
    assert delta_200 < 0.03
    # Lower number of projections
    nb_angles = 80
    radon_image_80 = radon(image, theta=np.linspace(0, 180, nb_angles,
                    endpoint=False))
    # Test whether the sum of all projections is approximately the same
    s = radon_image_80.sum(axis=0)
    assert np.allclose(s, s[0], rtol=0.01)
    reconstructed = iradon(radon_image_80)
    delta_80 = np.mean(abs(image/np.max(image) - reconstructed/np.max(reconstructed)))
    # Loss of quality when the number of projections is reduced
    assert delta_80 > delta_200

        
if __name__ == "__main__":
    run_module_suite()

