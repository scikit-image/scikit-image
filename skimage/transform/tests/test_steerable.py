import numpy as np
from numpy.testing import (assert_raises, assert_, assert_equal,
                           run_module_suite)
from skimage.filters.steerable import *

class TestSteerable(object):

	def testDifferentSize(self):
		im = np.random.randint(0, 255, (128, 128))
		a = Steerable()
		coeff = a.buildSCFpyr(im)
		out = a.reconSCFpyr(coeff)

		assert_equal(np.allclose(out, im, atol = 10), True)

if __name__ == "__main__":
	run_module_suite()
