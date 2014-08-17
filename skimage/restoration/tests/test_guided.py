import numpy as np
from numpy.testing import (assert_allclose, assert_raises, assert_equal)
from skimage.restoration._guided_filter import (_guided_filter,
                                                _guided_filter_same,
                                                guided_filter)


class TestGuidedFilter(object):

    def setup(self):
        np.random.seed(1111)
        self.image = np.zeros((100, 100))
        self.image[25:75, 25:75] = 1
        self.noise = np.random.normal(scale=0.2, size=self.image.shape)
        self.multichannel = np.tile(self.image[..., np.newaxis], (1, 1, 3))

    def test_self_guided(self):
        """Test that guiding an image with itself is within rounding error"""
        guided = _guided_filter(self.image, self.image, 0.1, 5)
        guided_self = _guided_filter_same(self.image, 0.1, 5)
        assert_allclose(guided, guided_self, rtol=0, atol=1e-10)

    def test_smoothing(self):
        """Test that increasing eta reduces variance in image."""
        etas = [0.01, 0.1, 1]
        var_baseline = self.noise.var()
        variances = np.array([_guided_filter_same(self.noise, eta, 5).var()
                              for eta in etas])
        assert (variances < var_baseline).all()

    def test_single_self(self):
        """Test guided_filter works for a single channel image with itself."""
        _guided = _guided_filter_same(self.image, 0.1, 5)
        guided = guided_filter(self.image, 0.1, 5)
        assert_equal(_guided, guided)

    def test_multi_self(self):
        """Test guided_filter  works for a multi channel image with itself, and
        is equivalent to the single channel case in a layer."""
        _guided_multi = _guided_filter_same(self.multichannel[..., 0], 0.1, 5)
        guided_multi = guided_filter(self.multichannel, 0.1, 5)
        assert_equal(_guided_multi, guided_multi[..., 0])

    def test_single_guide(self):
        """Test guided filter api works for a single channel image and a single
        channel guide."""
        _guided = _guided_filter(self.image, self.image - 1, 0.1, 5)
        guided = guided_filter(self.image, 0.1, 5, guide=self.image - 1)
        assert_equal(_guided, guided)

    def test_multi_single_guide(self):
        """Test guided filter api works for a multi channel image and a single
        channel guide."""
        _guided_multi = _guided_filter(self.multichannel[..., 0],
                                       self.multichannel[..., 0] - 1, 0.1, 5)
        guided_multi = guided_filter(self.multichannel, 0.1, 5,
                                     guide=self.multichannel[..., 0] - 1)
        assert_equal(_guided_multi, guided_multi[..., 0])

    def test_multi_multi_guide(self):
        """Test guided filter api works for a multi channel image and a multi
        channel guide."""
        _guided_multi = _guided_filter(self.multichannel[..., 0],
                                       self.multichannel[..., 0] - 1, 0.1, 5)
        guided_multi = guided_filter(self.multichannel, 0.1, 5,
                                     guide=self.multichannel - 1)
        assert_equal(_guided_multi, guided_multi[..., 0])

    def test_too_many_dimensions(self):
        """Test error is raised for incorrect input dimensions."""
        one_d = np.zeros(10)
        four_d = np.zeros((2, 2, 2, 2))
        assert_raises(ValueError, guided_filter, one_d, 0.1, 5)
        assert_raises(ValueError, guided_filter, four_d, 0.1, 5)

    def test_incompatible_guide(self):
        """Test error is raised for incompatible channels in image and
        guide."""
        assert_raises(ValueError, guided_filter, self.multichannel,
                      0.1, 5, self.multichannel[..., :2])


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
