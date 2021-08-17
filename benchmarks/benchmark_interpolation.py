# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import transform


class InterpolationResize:

    param_names = ['new_shape', 'order', 'mode', 'dtype', 'anti_aliasing']
    params = [
        ((500, 800), (2000, 4000), (80, 80, 80), (150, 150, 150)),  # new_shape
        (0, 1, 3, 5),    # order
        ('symmetric',),  # mode
        (np.float64, ),  # dtype
        (True,),         # anti_aliasing
    ]

    """Benchmark for filter routines in scikit-image."""
    def setup(self, new_shape, order, mode, dtype, anti_aliasing):
        ndim = len(new_shape)
        if ndim == 2:
            image = np.random.random((1000, 1000))
        else:
            image = np.random.random((100, 100, 100))
        self.image = image.astype(dtype, copy=False)

    def time_resize(self, new_shape, order, mode, dtype, anti_aliasing):
        transform.resize(self.image, new_shape, order=order, mode=mode,
                         anti_aliasing=anti_aliasing)

    def time_rescale(self, new_shape, order, mode, dtype, anti_aliasing):
        scale = tuple(s2 / s1 for s2, s1 in zip(new_shape, self.image.shape))
        transform.rescale(self.image, scale, order=order, mode=mode,
                          anti_aliasing=anti_aliasing)

    def peakmem_resize(self, new_shape, order, mode, dtype, anti_aliasing):
        transform.resize(self.image, new_shape, order=order, mode=mode,
                         anti_aliasing=anti_aliasing)

    def peakmem_reference(self, *args):
        """Provide reference for memory measurement with empty benchmark.

        Peakmem benchmarks measure the maximum amount of RAM used by a
        function. However, this maximum also includes the memory used
        during the setup routine (as of asv 0.2.1; see [1]_).
        Measuring an empty peakmem function might allow us to disambiguate
        between the memory used by setup and the memory used by target (see
        other ``peakmem_`` functions below).

        References
        ----------
        .. [1]: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#peak-memory
        """  # noqa
        pass
