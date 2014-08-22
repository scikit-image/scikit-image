import numpy as np
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern


class TestGLCM():

    def setup(self):
        self.image = np.array([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [0, 2, 2, 2],
                               [2, 2, 3, 3]], dtype=np.uint8)

    def test_output_angles(self):
        result = greycomatrix(self.image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 4)
        assert result.shape == (4, 4, 1, 4)
        expected1 = np.array([[2, 2, 1, 0],
                             [0, 2, 0, 0],
                             [0, 0, 3, 1],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
        expected2 = np.array([[1, 1, 3, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 1], expected2)
        expected3 = np.array([[3, 0, 2, 0],
                             [0, 2, 2, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 2], expected3)
        expected4 = np.array([[2, 0, 0, 0],
                             [1, 1, 2, 0],
                             [0, 0, 2, 1],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 3], expected4)

    def test_output_symmetric_1(self):
        result = greycomatrix(self.image, [1], [np.pi / 2], 4,
                              symmetric=True)
        assert result.shape == (4, 4, 1, 1)
        expected = np.array([[6, 0, 2, 0],
                             [0, 4, 2, 0],
                             [2, 2, 2, 2],
                             [0, 0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_output_distance(self):
        im = np.array([[0, 0, 0, 0],
                       [1, 0, 0, 1],
                       [2, 0, 0, 2],
                       [3, 0, 0, 3]], dtype=np.uint8)
        result = greycomatrix(im, [3], [0], 4, symmetric=False)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_output_combo(self):
        im = np.array([[0],
                       [1],
                       [2],
                       [3]], dtype=np.uint8)
        result = greycomatrix(im, [1, 2], [0, np.pi / 2], 4)
        assert result.shape == (4, 4, 2, 2)

        z = np.zeros((4, 4), dtype=np.uint32)
        e1 = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]], dtype=np.uint32)
        e2 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=np.uint32)

        np.testing.assert_array_equal(result[:, :, 0, 0], z)
        np.testing.assert_array_equal(result[:, :, 1, 0], z)
        np.testing.assert_array_equal(result[:, :, 0, 1], e1)
        np.testing.assert_array_equal(result[:, :, 1, 1], e2)

    def test_output_empty(self):
        result = greycomatrix(self.image, [10], [0], 4)
        np.testing.assert_array_equal(result[:, :, 0, 0],
                                      np.zeros((4, 4), dtype=np.uint32))
        result = greycomatrix(self.image, [10], [0], 4, normed=True)
        np.testing.assert_array_equal(result[:, :, 0, 0],
                                      np.zeros((4, 4), dtype=np.uint32))

    def test_normed_symmetric(self):
        result = greycomatrix(self.image, [1, 2, 3],
                              [0, np.pi / 2, np.pi], 4,
                              normed=True, symmetric=True)
        for d in range(result.shape[2]):
            for a in range(result.shape[3]):
                np.testing.assert_almost_equal(result[:, :, d, a].sum(),
                                               1.0)
                np.testing.assert_array_equal(result[:, :, d, a],
                                              result[:, :, d, a].transpose())

    def test_contrast(self):
        result = greycomatrix(self.image, [1, 2], [0], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        contrast = greycoprops(result, 'contrast')
        np.testing.assert_almost_equal(contrast[0, 0], 0.586)

    def test_dissimilarity(self):
        result = greycomatrix(self.image, [1], [0, np.pi / 2], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = greycoprops(result, 'dissimilarity')
        np.testing.assert_almost_equal(dissimilarity[0, 0], 0.418)

    def test_dissimilarity_2(self):
        result = greycomatrix(self.image, [1, 3], [np.pi / 2], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = greycoprops(result, 'dissimilarity')[0, 0]
        np.testing.assert_almost_equal(dissimilarity, 0.664)

    def test_invalid_property(self):
        result = greycomatrix(self.image, [1], [0], 4)
        np.testing.assert_raises(ValueError, greycoprops,
                                 result, 'ABC')

    def test_homogeneity(self):
        result = greycomatrix(self.image, [1], [0, 6], 4, normed=True,
                              symmetric=True)
        homogeneity = greycoprops(result, 'homogeneity')[0, 0]
        np.testing.assert_almost_equal(homogeneity, 0.80833333)

    def test_energy(self):
        result = greycomatrix(self.image, [1], [0, 4], 4, normed=True,
                              symmetric=True)
        energy = greycoprops(result, 'energy')[0, 0]
        np.testing.assert_almost_equal(energy, 0.38188131)

    def test_correlation(self):
        result = greycomatrix(self.image, [1, 2], [0], 4, normed=True,
                              symmetric=True)
        energy = greycoprops(result, 'correlation')
        np.testing.assert_almost_equal(energy[0, 0], 0.71953255)
        np.testing.assert_almost_equal(energy[1, 0], 0.41176470)

    def test_uniform_properties(self):
        im = np.ones((4, 4), dtype=np.uint8)
        result = greycomatrix(im, [1, 2, 8], [0, np.pi / 2], 4, normed=True,
                              symmetric=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity',
                     'energy', 'correlation', 'ASM']:
            greycoprops(result, prop)


class TestLBP():

    def setup(self):
        self.image = np.array([[255,   6, 255,   0,  141,   0],
                               [ 48, 250, 204, 166,  223,  63],
                               [  8,   0, 159,  50,  255,  30],
                               [167, 255,  63,  40,  128, 255],
                               [  0, 255,  30,  34,  255,  24],
                               [146, 241, 255,   0,  189, 126]], dtype='double')

    def test_default(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'default')
        ref = np.array([[  0, 251,   0, 255,  96, 255],
                        [143,   0,  20, 153,  64,  56],
                        [238, 255,  12, 191,   0, 252],
                        [129,  64.,  62, 159, 199,   0],
                        [255,   4, 255, 175,   0, 254],
                        [  3,   5,   0, 255,   4,  24]])
        np.testing.assert_array_equal(lbp, ref)

    def test_ror(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'ror')
        ref = np.array([[  0, 127,   0, 255,   3, 255],
                        [ 31,   0,   5,  51,   1,   7],
                        [119, 255,   3, 127,   0,  63],
                        [  3,   1,  31,  63,  31,   0],
                        [255,   1, 255,  95,   0, 127],
                        [  3,   5,   0, 255,   1,   3]])
        np.testing.assert_array_equal(lbp, ref)

    def test_uniform(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'uniform')
        ref = np.array([[0, 7, 0, 8, 2, 8],
                        [5, 0, 9, 9, 1, 3],
                        [9, 8, 2, 7, 0, 6],
                        [2, 1, 5, 6, 5, 0],
                        [8, 1, 8, 9, 0, 7],
                        [2, 9, 0, 8, 1, 2]])
        np.testing.assert_array_equal(lbp, ref)

    def test_var(self):
        # Test idea: mean of variance is estimate of overall variance.

        # Fix random seed for test stability.
        np.random.seed(13141516)

        # Create random image with known variance.
        image = np.random.rand(500, 500)
        target_std = 0.3
        image = image / image.std() * target_std

        # Use P=4 to avoid interpolation effects
        P, R = 4, 1
        lbp = local_binary_pattern(image, P, R, 'var')

        # Take central part to avoid border effect.
        lbp = lbp[5:-5,5:-5]

        # The LBP variance is biased (ddof=0), correct for that.
        expected = target_std**2 * (P-1)/P

        np.testing.assert_almost_equal(lbp.mean(), expected, 4)

    def test_nri_uniform(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'nri_uniform')
        ref = np.array([[ 0, 54,  0, 57, 12, 57],
                        [34,  0, 58, 58,  3, 22],
                        [58, 57, 15, 50,  0, 47],
                        [10,  3, 40, 42, 35,  0],
                        [57,  7, 57, 58,  0, 56],
                        [ 9, 58,  0, 57,  7, 14]])
        np.testing.assert_array_almost_equal(lbp, ref)


if __name__ == '__main__':
    np.testing.run_module_suite()
