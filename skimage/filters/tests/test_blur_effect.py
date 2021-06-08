from skimage.data import astronaut
from skimage.filters import blur_effect, gaussian


def test_blur_effect():
    image = astronaut()
    # Test that the blur metric increases with more blurring
    B0, _ = blur_effect(image)
    B1, _ = blur_effect(gaussian(image, sigma=1))
    B2, _ = blur_effect(gaussian(image, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2
