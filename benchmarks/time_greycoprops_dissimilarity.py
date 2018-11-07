import numpy as np
from skimage.feature import greycomatrix, greycoprops


class GreyCoPropsDissimilarity:
    """Benchmark for the greycomatrix in scikit-image."""

    # All parameters combinations will be tests.
    params = [[(50, 50), (100, 100), (200, 200), (400, 400)],  # shape
              [True, False],           # symmetric
              [True, False],           # normed
              ]

    # These are friendly names that will appear on the graphs.
    param_names = ['shape', 'symmetric', 'normed']

    def setup(self, shape, symmetric, normed):
        # Unless you need random to show the performance of a
        # particular algorithm, it is probably fastest to
        # allocate the array as ``full``.
        # This ensures that the memory is directly available to
        # routine without continuously pagefaulting
        # in this case, I want to make sure that we are hitting all
        # combinations of distances in the covariance matrix.
        # self.image - np.full(shape, fill_value=1)
        # I use an instance RandomState here to provide repeatable
        # results
        prng = np.random.RandomState(1234567) # Your favourite seed here
        self.image = prng.randint(0, 256, shape, 'ubyte')
        self.glcm = greycomatrix(self.image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                 symmetric=symmetric, normed=normed)

    # You need to include the shape parameter even if you don't use it
    # in your function
    def time_greycoprops_dissimilarity(self, shape, symmetric, normed):
        for i in range(100):
            greycoprops(self.glcm, 'dissimilarity', normed=normed)