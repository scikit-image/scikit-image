import numpy as np


def find_bitdepth(image):
    """returns the max bith depth of a uint16 image
    """
    umax = np.max(image)
    if umax > 2:
        return int(np.log2(umax))
    else:
        return 1
