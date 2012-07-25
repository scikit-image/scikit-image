from skimage import data
from skimage import feature
from skimage import img_as_float

def test_histogram_of_oriented_gradients():
    img = img_as_float(data.lena()[:256, :].mean(axis=2))

    fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1))

    assert len(fd) == 9 * (256 // 8) * (512 // 8)

def test_hog_image_size_cell_size_mismatch():
    image = data.camera()[:150, :200]
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1))
    assert len(fd) == 9 * (150 // 8) * (200 // 8)

if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
