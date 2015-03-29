from skimage import transform
import numpy as np
from numpy import testing

def energy(img):
    if(img.ndim == 3):
        return np.ascontiguousarray(img[:, :, 0])
    return (1 - img)

def test_seam_carving():
    img = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0 , 0, 0, 0]], dtype = np.float )

    out = transform.seam_carve(img, 'horizontal', 1, energy, border=0)
    testing.assert_allclose(out, 0)

    img = img.T
    out = transform.seam_carve(img, 'vertical', 1, energy, border=0)
    testing.assert_allclose(out, 0)

    img = img.T

    img3 = np.dstack([img, img, img])

    out = transform.seam_carve(img3, 'horizontal', 1, energy, border=0)
    testing.assert_allclose(out, 0)


    out = transform.seam_carve(img3, 'vertical', 1, energy, border=0)
    testing.assert_allclose(out, 0)
