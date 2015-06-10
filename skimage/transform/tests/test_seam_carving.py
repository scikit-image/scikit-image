from skimage import transform
import numpy as np
from numpy import testing


def test_seam_carving():
    img = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]], dtype=np.float)
    energy = 1 - img

    out = transform.seam_carve(img, energy, 'vertical', 1, border=0)
    testing.assert_allclose(out, 0)

    img = img.T
    out = transform.seam_carve(img, energy, 'horizontal', 1, border=0)
    testing.assert_allclose(out, 0)


if __name__ == '__main__':
    np.testing.run_module_suite()
