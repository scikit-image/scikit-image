import numpy as np

from skimage import data, io
from skimage.filter.rank import median
from skimage.morphology import disk

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider, OKCancelButtons, SaveButtons
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.plugins import LineProfile

from skimage.viewer.qt import qt_api, QtCore, QtGui
from numpy.testing import assert_almost_equal
from numpy.testing.decorators import skipif


@skipif(qt_api is None)
def test_custom_plugin():

    image = data.coins()[:-50, :]  # shave some off to make the line lower
    viewer = ImageViewer(image)

    def median_filter(img, radius):
        return median(img, selem=disk(radius))

    plugin = Plugin(image_filter=median_filter)
    sld = Slider('radius', 2, 10, value_type='int')
    plugin += sld
    sv = SaveButtons()
    plugin += sv
    ok = OKCancelButtons()
    plugin += ok

    viewer += plugin

    lp = LineProfile()
    viewer += lp

    def validate_pre(image):
        line = lp.get_profiles()[-1][0]
        assert line.size == 129
        assert_almost_equal(np.std(image), 49.998, 3)
        assert_almost_equal(np.std(line), 56.012, 3)
        assert_almost_equal(np.max(line) - np.min(line), 159.0, 1)

    def validate_post(image):
        line = lp.get_profiles()[-1][0]
        assert_almost_equal(np.std(image), 51.364, 3)
        assert_almost_equal(np.std(line), 56.3555, 3)
        assert_almost_equal(np.max(line) - np.min(line), 172.0, 1)

    assert plugin in viewer.plugins
    assert lp in viewer.plugins

    validate_pre(viewer.image)

    timer = QtCore.QTimer()
    timer.singleShot(1000, lambda: setattr(sld, 'val', 1))
    timer.singleShot(1200, lambda: validate_post(viewer.image))
    timer.singleShot(1300, lambda: ok.update_original_image())
    timer.singleShot(1400, lambda: sv.save_to_stack())
    timer.singleShot(1500, lambda: viewer.close())
    timer.singleShot(1600, lambda: QtGui.QApplication.quit())

    viewer.show()

    validate_post(viewer.image)
    img = io.pop()
    validate_post(img)
