import numpy as np
from skimage.transform import SimilarityTransform, warp
import warnings
import functools
import inspect

try:
    from skimage.util.dtype import _convert as convert
except ImportError:
    from skimage.util.dtype import convert


class WarpSuite:
    params = ([np.uint8, np.uint16, np.float32, np.float64],
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
        self.tform = SimilarityTransform(scale=1, rotation=np.pi / 10,
                                         translation=(0, 4))
        self.tform.params = self.tform.params.astype('float32')
        self.order = order

        if 'dtype' in inspect.signature(warp).parameters:
            self.warp = functools.partial(warp, dtype=self.image.dtype)
        else:
            # Keep a call to functools to have the same number of python
            # function calls
            self.warp = functools.partial(warp)

    # def time_same_type(self, dtype_in, N, order, dtype_tform):
    def time_same_type(self, dtype_in, N, order):
        """Test the case where the users wants to preserve their same low
        precision data type."""
        result = self.warp(self.image, self.tform, order=self.order,
                           preserve_range=True)

        # convert back to input type, no-op if same type
        result = result.astype(dtype_in, copy=False)

    # def time_to_float64(self, dtype_in, N, order, dtype_form):
    def time_to_float64(self, dtype_in, N, order):
        """Test the case where want to upvert to float64 for continued
        transformations."""
        result = warp(self.image, self.tform, order=self.order,
                      preserve_range=True)
