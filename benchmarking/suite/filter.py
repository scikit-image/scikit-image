from vbench.api import Benchmark

common_setup ="""from vb_common import *
"""

setup = common_setup + """
from skimage import filter
from skimage import data

image = data.camera()

"""

sobel = "filter.sobel(image)"
filter_sobel = Benchmark(sobel, setup, name="sobel_test")

