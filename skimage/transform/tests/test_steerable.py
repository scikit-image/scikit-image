from numpy.testing import assert_array_almost_equal, assert_raises, run_module_suite
from skimage.transform import steerable
import numpy as np
from skimage import img_as_float

def test_steerable_reconstruction():
	im = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
	coeff = steerable.build_steerable(im)
	out = steerable.recon_steerable(coeff)

	assert_array_almost_equal(img_as_float(im), out, decimal = 2)

def test_steerable_reconstruction_shape():
	im = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
	coeff = steerable.build_steerable(im)
	out = steerable.recon_steerable(coeff)

	assert_array_almost_equal(img_as_float(im), out, decimal = 2)

# def test_steerable_reconstruction():
# 	im = np.random.randint(0, 255, (113, 29), dtype=np.uint8)
# 	coeff = steerable.build_steerable(im)
# 	out = steerable.recon_steerable(coeff)

# 	print(coeff[0].shape)
# 	print(coeff[1][0].shape)
# 	print(coeff[2][0].shape)

# 	assert_array_almost_equal(im, out, decimal = 0)


if __name__ == "__main__":
	run_module_suite()
