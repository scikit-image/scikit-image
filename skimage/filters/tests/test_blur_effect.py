from skimage.data import astronaut
from skimage.filters import blur_effect, gaussian

image = astronaut()


def test_blur_effect():
    # Test that the blur metric increases with more blurring
    B0, _ = blur_effect(image)
    B1, _ = blur_effect(gaussian(image, sigma=1))
    B2, _ = blur_effect(gaussian(image, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


def test_blur_effect_hsize():
    # Test that the blur metric decreases with increasing size of the
    # re-blurring filter
    B0, _ = blur_effect(image, h_size=3)
    B1, _ = blur_effect(image)  # default h_size is 11
    B2, _ = blur_effect(image, h_size=30)
    assert 0 <= B0 < 1
    assert B0 > B1 > B2
