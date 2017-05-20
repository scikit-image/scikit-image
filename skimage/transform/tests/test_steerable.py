from numpy.testing import assert_array_almost_equal, assert_raises, run_module_suite
from skimage.transform import steerable

def test_steerable_reconstruction():
	im = np.random.randint(0, 255, (128, 128))
	coeff = steerable.build_steerable(im)
	out = recon_steerable(coeff)

	assert_array_almost_equal(im, out, decimal = 0)

if __name__ == "__main__":
	run_module_suite()
