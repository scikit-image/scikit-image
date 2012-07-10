import numpy as np
from skimage.feature import greycomatrix, greycoprops


class TestGLCM():
    def setup(self):
        self.image = np.array([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [0, 2, 2, 2],
                               [2, 2, 3, 3]], dtype=np.uint8)

    def test_output_angles(self):
        result = greycomatrix(self.image, [1], [0, np.pi / 2], 4)
        assert result.shape == (4, 4, 1, 2)
        expected1 = np.array([[2, 2, 1, 0],
                             [0, 2, 0, 0],
                             [0, 0, 3, 1],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
        expected2 = np.array([[3, 0, 2, 0],
                             [0, 2, 2, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 1], expected2)

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

if __name__ == '__main__':
    np.testing.run_module_suite()
