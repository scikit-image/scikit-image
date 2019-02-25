from skimage import transform
import pytest


def test_seam_carving():
    with pytest.raises(NotImplementedError):
        transform.seam_carve()


if __name__ == '__main__':
    np.testing.run_module_suite()
