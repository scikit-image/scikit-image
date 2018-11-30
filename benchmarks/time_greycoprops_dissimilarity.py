import numpy as np
from skimage.feature import greycomatrix, greycoprops


class GreyCoPropsDissimilarity:
    """Benchmark for the greycomatrix in scikit-image."""

    # All parameters combinations will be tests.
    params = [[(50, 50), (100, 100), (200, 200), (400, 400)],  # shape
              [True, False],           # symmetric
              [True, False],           # pre_normalized
              ]

    # These are friendly names that will appear on the graphs.
    param_names = ['shape', 'symmetric', 'pre_normalized']

    def setup(self, shape, symmetric, pre_normalized):
        # I use an instance RandomState here to provide repeatable
        # results
        prng = np.random.RandomState(1234567) # Your favourite seed here
        self.image = prng.randint(0, 256, shape, np.uint8)
        self.glcm = greycomatrix(self.image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                 symmetric=symmetric, pre_normalized=pre_normalized)

    # You need to include the shape parameter even if you don't use it
    # in your function
    def time_greycoprops_dissimilarity(self, shape, symmetric, pre_normalized):
        for i in range(100):
            greycoprops(self.glcm, 'dissimilarity', pre_normalized=pre_normalized)