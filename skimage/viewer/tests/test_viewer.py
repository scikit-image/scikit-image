import skimage
import skimage.data as data
from skimage.viewer import ImageViewer
from skimage.viewer.qt import qt_api
from numpy.testing import assert_equal, assert_allclose
from numpy.testing.decorators import skipif


def setup_line_profile(image):
    from skimage.viewer.plugins.lineprofile import LineProfile
    viewer = ImageViewer(skimage.img_as_float(image))
    plugin = LineProfile()
    viewer += plugin
    return plugin


@skipif(qt_api is None)
def test_line_profile():
    """ Test a line profile using an ndim=2 image"""
    plugin = setup_line_profile(data.camera())
    line_image, scan_data = plugin.output()
    for inp in [line_image.nonzero()[0].size,
                line_image.sum() / line_image.max(),
                scan_data.size]:
        assert_equal(inp, 172)
    assert_equal(line_image.shape, (512, 512))
    assert_allclose(scan_data.max(), 0.9139, rtol=1e-3)
    assert_allclose(scan_data.mean(), 0.2828, rtol=1e-3)


@skipif(qt_api is None)
def test_line_profile_rgb():
    """ Test a line profile using an ndim=3 image"""
    plugin = setup_line_profile(data.chelsea())
    for i in range(6):
        plugin.line_tool._thicken_scan_line()
    line_image, scan_data = plugin.output()
    assert_equal(line_image[line_image == 128].size, 755)
    assert_equal(line_image[line_image == 255].size, 151)
    assert_equal(line_image.shape, (300, 451))
    assert_equal(scan_data.shape, (152, 3))
    assert_allclose(scan_data.max(), 0.772, rtol=1e-3)
    assert_allclose(scan_data.mean(), 0.4355, rtol=1e-3)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
