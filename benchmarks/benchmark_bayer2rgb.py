from skimage import data
from skimage.color.bayer2rgb import implementations

import numpy as np


class Bayer2rgbSuite:
    param_names = ["shape", "dtype_in", "dtype_out", "implementation"]

    # Typically, the input is a uint8, and users will likely want a
    # uint8 output or float64 output.
    # 128 is shown to accurately show the discrepancies between the
    # implementations in terms of timing.
    # 4096x4096 shows how the algorithm behaves out of cache
    #
    # on my laptop, images of size less than 4096x4096 appeared to behave
    # as "in-cache" images.

    params = [[(128, 128), (4096, 4096)],
              ["uint8", "float64"],
              ["uint8", "float64"],
              implementations.keys()]

    def setup(self, shape, dtype_in, dtype_out, implementation):
        dtype_in = np.dtype(dtype_in)
        if dtype_in.kind in 'ui' :
            self.image = np.random.randint(0, 255, size=shape, dtype=dtype_in)
        else:
            self.image = np.random.random(shape).astype(dtype_in)
        self.bayer2rgb = implementations[implementation]

    def time_bayer2rgb(self, shape, dtype_in, dtype_out, implementation):
        self.bayer2rgb(self.image, dtype=dtype_out)


"""
'''
This benchmark requires opencv.
Not really worth adding as a bench dependency since we can't even compare in
terms of performance
· Discovering benchmarks
· Running 1 total benchmarks (1 commits * 1 environments * 1 benchmarks)
[  0.00%] ·· Benchmarking existing-py_home_mark2_miniconda3_envs...
[ 50.00%] ··· benchmark_bayer2rgb.Bayer2rgbOpenCVSuite.time_bayer2rgb
[ 50.00%] ··· ============== ========= ==========
              --                implementation
              -------------- --------------------
                  shape       skimage    opencv
              ============== ========= ==========
                (128, 128)    694±0μs   215±0μs
               (4096, 4096)   349±0ms   53.0±0ms
              ============== ========= ==========
'''
import cv2
from skimage.color import bayer2rgb

class Bayer2rgbOpenCVSuite:
    param_names = ["shape", "implementation"]
    params = [[(128, 128), (4096, 4096)],
              ['skimage', 'opencv']]

    def setup(self, shape, implementation):
        self.image = np.random.randint(0, 255, size=shape, dtype='uint8')

    def time_bayer2rgb(self, shape, implementation):
        if implementation == 'skimage':
            bayer2rgb(self.image, dtype='uint8')
        elif implementation == 'opencv':
            # The actual pattern doesnt' matter much
            cv2.cvtColor(self.image, cv2.COLOR_BAYER_BG2RGB)
"""
