import numpy as np
from skimage.transform import SimilarityTransform, warp
from skimage.util.dtype import convert


class WarpSuite:
    params = ([np.uint8, np.uint16, np.float32, np.float64],
              [128, 1024, 4096],
              [0, 1, 3]
              )
    param_names = ['dtype_in', 'N', 'order']

    def setup(self, dtype_in, N, order):
        self.image = convert(np.random.random((N, N)), dtype=dtype_in)
        self.tform = SimilarityTransform(scale=1, rotation=np.pi / 10,
                                         translation=(0, 4))
        self.order = order

    def time_same_type(self):
        """Test the case where the users wants to preserve their same low
        precision data type."""
        result = warp(self.image, self.tform, order=self.order,
                      preserve_range=True)  # , dtype=self.image.dtype)

        # With PR #3253, this line will be unecessary, and we can pass
        # the parameter dtype_out above to specify casting back to the
        # same time
        result = result.astype(self.image.dtype)

    def time_to_float64(self):
        """Test the case where want to upvert to float64 for continued
        transformations."""
        result = warp(self.image, self.tform, order=self.order,
                      preserve_range=True)  # , dtype=np.float64)
