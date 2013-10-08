import numpy as np
import unittest
from skimage.filter._wavelet import wavelet_filter, wavelet_coefficient_array
from skimage.filter._wavelet import wavelet_list, bayes_shrink, visu_shrink


class TestWaveletFilter(unittest.TestCase):

    def test_wavelet_names(self):
        """
        Number of supported wavelets could conceivably increase, don't
        want to hardcode them or the test
        """
        wavelist = wavelet_list()
        assert len(wavelist) >= 76
        assert "haar" in wavelist

    def test_filter_null(self):
        """
        DWT is orthogonal -> takes null matrix to null matrix
        """
        a = np.zeros((4, 4))
        assert np.all((wavelet_filter(a, 1) == a))

    def test_filter_bad_thresholds(self):
        """
        Improperly stated thresholds should raise error
        """
        message = "Wavelet threshold values not set correctly."
        a = np.ones((12, 12))

        t = [0., [1.]]
        with self.assertRaises(Exception) as context:
            wavelet_filter(a, t)
        self.assertEqual(context.exception.message, message)

        t = [0., 3.]
        with self.assertRaises(Exception) as context:
            wavelet_filter(a, t, level=3)
        self.assertEqual(context.exception.message, message)

        t = [[0., 3.]]
        with self.assertRaises(Exception) as context:
            wavelet_filter(a, t, level=1)
        self.assertEqual(context.exception.message, message)

    def test_filter_good_thresholds(self):
        """
        Properly formed thresholds should pass.

        Also verifies that wavelet coefficient thresholding is monotonic
        w.r.t norms, std, etc.
        as you proceed through wavelet decomposition levels.
        """
        a = np.random.randn(10, 10)
        s = a.std()

        t = 10.
        bs = wavelet_filter(a, t, level=1).std()
        assert s > bs

        t = [10., 10.]
        cs = wavelet_filter(a, t, level=2).std()
        assert s > bs > cs

        t = [[10., 10., 10.], [10., 10., 10.], [1., 2., 3.]]
        ds = wavelet_filter(a, t, level=3).std()
        assert s > bs > cs > ds

    def test_coefficient_array(self):
        """
        Tests coefficient array on small test case
        """
        a = np.array([[1, 0], [0, 1]])
        b = wavelet_coefficient_array(a)
        assert np.all(np.array([[255., 0], [0, 0]]) == b)

    def test_coefficient_array_odd(self):
        """
        Tests coefficient array on larger, odd-dimensioned case
        """
        a = np.zeros((5, 3))
        a[:2, :3] = 1.
        b = wavelet_coefficient_array(a)
        c = np.zeros((5, 3))
        c[0, 0:2] = 255.
        c[1, 0:2] = 76.
        assert np.all(np.array(c == b))

    def test_bayesshrink(self):
        a = np.random.randn(10, 10)
        s = a.std()
        bs = bayes_shrink(a, level=2).std()
        assert s > bs

    def test_visushrink(self):
        a = np.random.randn(10, 10)
        s = a.std()
        bs = visu_shrink(a, level=2).std()
        assert s > bs

if __name__ == '__main__':
    unittest.main()
