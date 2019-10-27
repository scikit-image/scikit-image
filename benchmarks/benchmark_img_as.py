import numpy as np
from skimage.util.dtype import convert


class ImgAsSuite:
    param_names = ["dtype_in", "dtype_out", "shape", "mode"]
    params = [
        [np.uint8, np.uint16, np.float32, np.float64],
        [np.uint8, np.uint16, np.float32, np.float64],
        [
            (32, 32),
            (256, 256),
            (1024, 1024),
            (4096, 4096),
        ],
        ["library", "unsafe"]
    ]

    def setup(self, dtype_in, dtype_out, shape, mode):
        if np.issubdtype(dtype_in, np.floating):
            fill_value = 0.2
        else:
            fill_value = 51
        self.image = np.full(shape, fill_value=fill_value, dtype=dtype_in)

    def _time_noop(self, *args):
        # As array isn't as cheap as a "pass", and therefore, I want to time
        # The cost of this operation.
        np.asarray(self.image)

    def time_img_as_(self, dtype_in, dtype_out, shape, mode):
        if mode == "library":
            convert(self.image, dtype_out)
        else:
            self.img_as_manual(dtype_in, dtype_out)

    def img_as_manual(self, dtype_in, dtype_out):
        # naive implementations of img_as to compare the performance of the
        # library
        if dtype_in == dtype_out:
            return np.asarray(self.image)

        elif (np.issubdtype(dtype_in, np.integer)
                and np.issubdtype(dtype_out, np.integer)):

            max_in = np.iinfo(dtype_in).max
            max_out = np.iinfo(dtype_out).max

            result_float = np.multiply(self.image, max_out / max_in)
            return result_float.astype(dtype_out)

        elif (np.issubdtype(dtype_in, np.integer)
                and np.issubdtype(dtype_out, np.floating)):
            imax = np.iinfo(dtype_in).max
            return np.multiply(self.image, 1 / imax, dtype=dtype_out)

        elif (np.issubdtype(dtype_in, np.floating)
                and np.issubdtype(dtype_out, np.floating)):
            return self.image.astype(dtype_out)
        elif (np.issubdtype(dtype_in, np.floating)
                and np.issubdtype(dtype_out, np.integer)):
            imax = np.iinfo(dtype_out).max
            # Add 1/(imax * 2) to implement the correct rounding
            return np.multiply(
                self.image  + (1 / (imax * 2)), imax).astype(dtype_out)

        raise NotImplementedError()
