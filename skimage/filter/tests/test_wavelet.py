import numpy as np
import unittest
from skimage.filter._wavelet import wavelet_filter, wavelet_coefficient_array
from skimage.filter._wavelet import wavelet_list, bayes_shrink, visu_shrink

try:
    from scipy.misc import imresize
except:
    imresize = False


class TestWaveletFilter(unittest.TestCase):

    def test_wavelet_names(self):
        """
        Tests wavelet_list().
        Number of supported wavelets could conceivably increase, so don't
        want to hardcode a fixed list into the test
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
        self.assertEqual(str(context.exception), message)

        t = [0., 3.]
        with self.assertRaises(Exception) as context:
            wavelet_filter(a, t, level=3)
        self.assertEqual(str(context.exception), message)

        t = [[0., 3.]]
        with self.assertRaises(Exception) as context:
            wavelet_filter(a, t, level=1)
        self.assertEqual(str(context.exception), message)

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
        a = np.eye(10)
        b = wavelet_coefficient_array(a)
        np.testing.assert_array_almost_equal(np.diag(b), np.ones(10))

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
