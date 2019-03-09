import numpy as np
from skimage.transform import SimilarityTransform, warp
from skimage.util.dtype import convert
import warnings
import functools
import inspect


class WarpSuite:
    params = ([np.float32, np.float64],
              [128, 1024, 4096],
              [0, 1, 3],
              # [np.float32, np.float64]
              )
    # param_names = ['dtype_in', 'N', 'order', 'dtype_tform']
    param_names = ['dtype_in', 'N', 'order']

    # def setup(self, dtype_in, N, order, dtype_tform):
    def setup(self, dtype_in, N, order):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Possible precision loss")
            self.image = convert(np.random.random((N, N)), dtype=dtype_in)
        tform = SimilarityTransform(scale=1, rotation=np.pi / 10,
                                    translation=(0, 4))
        tform.params = tform.params.astype('float32')
        order = order

        self.warp = functools.partial(
            warp, inverse_map=tform,
            order=order, preserve_range=True)

    # def time_same_type(self, dtype_in, N, order, dtype_tform):
    def time_warp(self, dtype_in, N, order):
        """Test the case where the users wants to preserve their same low
        precision data type."""
        result = self.warp(self.image)

        # convert back to input type, no-op if same type
        result = result.astype(dtype_in, copy=False)

