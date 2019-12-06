from skimage._shared.utils import check_nD
import numpy.testing as npt
import numpy as np
from skimage._shared import testing


def test_check_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        check_nD(x, 2)


if __name__ == "__main__":
    npt.run_module_suite()
