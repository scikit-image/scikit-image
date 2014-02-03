import skimage.data as data
from skimage.viewer import ImageViewer
from numpy.testing import assert_equal, assert_allclose


def setup_line_profile(image):
    from skimage.viewer.plugins.lineprofile import LineProfile
    viewer = ImageViewer(image)
    plugin = LineProfile()
    viewer += plugin
    return plugin


def test_line_profile():
    """ Test a line profile using an ndim=2 image"""
    plugin = setup_line_profile(data.camera())
    line_image, scan_data = plugin.output()
    for inp in [line_image.nonzero()[0].size,
                line_image.sum() / line_image.max(),
                scan_data.size]:
        assert_equal(inp, 172)
    assert_equal(line_image.shape, (512, 512))
    assert_equal(scan_data.max(), 234.0)
    assert_allclose(scan_data.mean(), 71.726744186046517)


def test_line_profile_rgb():
    """ Test a line profile using an ndim=3 image"""
    plugin = setup_line_profile(data.chelsea())
    for i in range(6):
        plugin.line_tool._thicken_scan_line()
    line_image, scan_data = plugin.output()
    assert_equal(line_image[line_image == 128].size, 906)
    assert_equal(line_image[line_image == 255].size, 151)
    assert_equal(line_image.shape, (300, 451))
    assert_equal(scan_data.shape, (151, 3))
    assert_allclose(scan_data.max(), 196.85714285714286)
    assert_allclose(scan_data.mean(), 111.17029328287606)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
